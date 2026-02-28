"""
SAKURA ORACLE — 組合せオッズスクレイパー

netkeibaのJSON APIから馬連・ワイド・馬単・三連複・三連単のオッズを取得する。

使い方:
    PYTHONIOENCODING=utf-8 py ml/scraper/odds_scraper.py 202609010411
"""

import pickle
import re
import sys
import time
import random
from pathlib import Path

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.scraper.config import (
    HEADERS, CACHE_DIR,
    REQUEST_TIMEOUT, MAX_RETRIES, MIN_WAIT, MAX_WAIT, BACKOFF_BASE,
)


# APIのオッズタイプ番号 → 馬券名
ODDS_TYPES: dict[str, str] = {
    "4": "馬連",
    "5": "ワイド",
    "6": "馬単",
    "7": "三連複",
    "8": "三連単",
}

# 馬券タイプごとのキー桁数（2桁×頭数）
KEY_DIGITS: dict[str, int] = {
    "馬連": 4,    # "0102" → [1, 2]
    "ワイド": 4,
    "馬単": 4,
    "三連複": 6,  # "010203" → [1, 2, 3]
    "三連単": 6,
}


def _polite_sleep() -> None:
    """netkeiba用の礼儀正しい待機（3-7秒ランダム）"""
    time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))


def _parse_key_to_horses(key: str, bet_type: str) -> list[int]:
    """APIのキー文字列から馬番リストを取得する。

    "0102" → [1, 2]
    "010203" → [1, 2, 3]
    """
    expected_len = KEY_DIGITS.get(bet_type, 4)
    if len(key) != expected_len:
        return []

    horses: list[int] = []
    for i in range(0, len(key), 2):
        num_str = key[i:i+2]
        try:
            num = int(num_str)
            if 1 <= num <= 18:
                horses.append(num)
        except ValueError:
            return []
    return horses


def _fetch_odds_api(race_id: str, odds_type: str) -> dict | None:
    """netkeibaのオッズAPIを呼び出す。

    Args:
        race_id: netkeibaのrace_id（12桁）
        odds_type: オッズタイプ番号（"4"=馬連, "5"=ワイド, ...）

    Returns:
        APIレスポンスのJSONデータ or None
    """
    # キャッシュチェック
    cache_key = f"odds_api_{race_id}_{odds_type}"
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    url = "https://race.netkeiba.com/api/api_get_jra_odds.html"
    params = {
        "race_id": race_id,
        "type": odds_type,
        "action": "update",
    }
    # APIリクエスト用のヘッダー（Referer, X-Requested-With追加）
    api_headers = {
        **HEADERS,
        "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b{odds_type}",
        "X-Requested-With": "XMLHttpRequest",
    }

    for attempt in range(MAX_RETRIES):
        try:
            _polite_sleep()
            r = requests.get(
                url, headers=api_headers, params=params,
                timeout=REQUEST_TIMEOUT, verify=False,
            )

            if r.status_code == 400:
                backoff = BACKOFF_BASE * (attempt + 1)
                print(f"  ⚠️ 400エラー。{backoff}秒待機...")
                time.sleep(backoff)
                continue
            if r.status_code == 404:
                print(f"  ⚠️ 404: APIエンドポイント")
                return None

            r.raise_for_status()
            data = r.json()

            # キャッシュ保存
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)

            return data

        except requests.RequestException as e:
            print(f"  リトライ {attempt + 1}/{MAX_RETRIES}: {e}")
            time.sleep(30)
        except ValueError as e:
            print(f"  JSON解析失敗: {e}")
            return None

    print(f"  ❌ API取得失敗: type={odds_type}")
    return None


def _parse_api_odds(api_data: dict, odds_type: str, bet_type: str) -> list[dict]:
    """APIレスポンスからオッズデータを抽出する。

    APIの応答形式:
        {"status": "middle", "data": {"odds": {"4": {"0102": ["472.9", "", "69"], ...}}}}

    ワイドの場合:
        {"0102": ["112.4", "116.0", "71"]}  → [下限, 上限, 人気順] → 平均値を使用

    Returns:
        [{"type": "馬連", "horses": [1, 2], "odds": 472.9}, ...]
    """
    odds_dict = api_data.get("data", {}).get("odds", {}).get(odds_type, {})
    if not odds_dict:
        return []

    results: list[dict] = []

    for key, values in odds_dict.items():
        if not isinstance(values, list) or len(values) < 1:
            continue

        # 馬番を解析
        horses = _parse_key_to_horses(key, bet_type)
        if not horses:
            continue

        # オッズ値を解析
        try:
            if bet_type == "ワイド" and len(values) >= 2 and values[1]:
                # ワイドはレンジ表記 → 平均値
                low = float(values[0])
                high = float(values[1])
                odds = (low + high) / 2.0
            else:
                odds = float(values[0])
        except (ValueError, TypeError, IndexError):
            continue

        if odds <= 0:
            continue

        results.append({
            "type": bet_type,
            "horses": horses,
            "odds": round(odds, 1),
        })

    return results


def scrape_combo_odds(race_id: str) -> list[dict]:
    """全馬券タイプのオッズを一括取得する。

    netkeibaのJSON APIを使用してオッズデータを取得する。

    Args:
        race_id: netkeibaのrace_id（12桁）

    Returns:
        [{"type": "馬連", "horses": [3, 12], "odds": 201.1}, ...]
        combo_ev.py の _load_excel_odds() と同じ出力形式。
    """
    print(f"\n--- 組合せオッズ取得 (race_id={race_id}) ---")
    all_results: list[dict] = []

    for odds_type, bet_type in ODDS_TYPES.items():
        print(f"  📊 {bet_type} (type={odds_type}) 取得中...")

        api_data = _fetch_odds_api(race_id, odds_type)
        if api_data is None:
            print(f"     → 取得失敗")
            continue

        # ステータス確認
        status = api_data.get("status", "")
        if status == "nodata":
            print(f"     → オッズ未公開（nodata）")
            continue

        results = _parse_api_odds(api_data, odds_type, bet_type)

        print(f"     → {len(results)}件")
        all_results.extend(results)

    print(f"\n  オッズ取得合計: {len(all_results)}件")
    return all_results


# フロントエンドの comboKey プレフィックス
TYPE_TO_PREFIX: dict[str, str] = {
    "馬連": "quinella",
    "ワイド": "wide",
    "三連複": "trio",
}


def to_combo_odds_map(odds_list: list[dict]) -> dict[str, float]:
    """スクレイピング結果をcomboKey→oddsマップに変換（フロントエンド用）。

    馬単/三連単はフロントのBOX/軸流しで未使用のためスキップ。

    Args:
        odds_list: scrape_combo_odds() の戻り値
            [{"type": "馬連", "horses": [5, 12], "odds": 10.6}, ...]

    Returns:
        {"quinella-5-12": 10.6, "wide-5-12": 114.2, "trio-5-9-12": 1858.3, ...}
    """
    result: dict[str, float] = {}
    for entry in odds_list:
        prefix = TYPE_TO_PREFIX.get(entry["type"])
        if prefix is None:
            # 馬単・三連単はフロントで未使用 → スキップ
            continue
        # 馬番をソートしてキー生成（フロントと同じ形式）
        nums = sorted(entry["horses"])
        key = f"{prefix}-{'-'.join(str(n) for n in nums)}"
        result[key] = entry["odds"]
    return result


def main() -> None:
    """CLI エントリーポイント。"""
    if len(sys.argv) < 2:
        print("使い方: PYTHONIOENCODING=utf-8 py ml/scraper/odds_scraper.py <race_id>")
        print()
        print("例:")
        print("  py ml/scraper/odds_scraper.py 202609010411")
        sys.exit(1)

    race_id = sys.argv[1]
    results = scrape_combo_odds(race_id)

    if results:
        # 馬券タイプ別の集計
        type_counts: dict[str, int] = {}
        for entry in results:
            type_counts[entry["type"]] = type_counts.get(entry["type"], 0) + 1
        print(f"\n  📋 馬券タイプ別:")
        for t, c in type_counts.items():
            print(f"    {t}: {c}件")

        print(f"\n  🏇 取得結果サンプル（先頭10件）:")
        print(f"  {'券種':>6} {'組合せ':>12} {'オッズ':>8}")
        print("  " + "-" * 30)
        for entry in results[:10]:
            targets = "-".join(str(h) for h in entry["horses"])
            print(f"  {entry['type']:>6} {targets:>12} {entry['odds']:>8.1f}")
        print(f"\n  合計: {len(results)}件")
    else:
        print("\n  ⚠️ オッズを取得できませんでした")
        print("  原因候補:")
        print("    - レースが未発走でオッズ未公開")
        print("    - race_id が不正")


if __name__ == "__main__":
    main()
