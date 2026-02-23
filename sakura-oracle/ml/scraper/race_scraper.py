"""
SAKURA ORACLE — netkeibaスクレイパー（改良版）

使い方:
    PYTHONIOENCODING=utf-8 py ml/scraper/race_scraper.py

処理フロー:
    1. 開催日ページ → レースリンク一覧 → キーワードで race_id 特定
    2. race_id → race.netkeiba.com（上がり3F・通過順あり）を優先取得
    3. 取得失敗時は db.netkeiba.com にフォールバック
    4. 全結果を CSV に統合出力
"""

import time
import random
import pickle
import re
import sys
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.config import (
    HEADERS, CACHE_DIR, RAW_DIR,
    TARGET_RACES,
    REQUEST_TIMEOUT, MAX_RETRIES, MIN_WAIT, MAX_WAIT, BACKOFF_BASE,
)


def polite_sleep() -> None:
    """netkeiba用の礼儀正しい待機（3-7秒ランダム）"""
    time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))


def safe_request(url: str, max_retries: int = MAX_RETRIES) -> bytes | None:
    """キャッシュ付き安全リクエスト"""
    cache_key = re.sub(r'[^a-zA-Z0-9]', '_', url)[-100:]
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    for attempt in range(max_retries):
        try:
            polite_sleep()
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, verify=False)

            if r.status_code == 400:
                backoff = BACKOFF_BASE * (attempt + 1)
                print(f"  ⚠️ 400エラー。{backoff}秒待機...")
                time.sleep(backoff)
                continue
            if r.status_code == 404:
                print(f"  ⚠️ 404: {url}")
                return None

            r.raise_for_status()
            r.encoding = r.apparent_encoding or "euc-jp"

            with open(cache_file, "wb") as f:
                pickle.dump(r.content, f)
            return r.content

        except requests.RequestException as e:
            print(f"  リトライ {attempt + 1}/{max_retries}: {e}")
            time.sleep(30)

    print(f"  ❌ 取得失敗: {url}")
    return None


# 手動フォールバック: 日付検索で見つからないレースのrace_id
MANUAL_RACE_IDS: dict[str, str] = {
    # 必要に応じて "label": "race_id" で追加
    # 例: "フェアリーS2025": "202506010511",
}


def find_race_id_from_date(date: str, keyword: str, label: str = "") -> str | None:
    """開催日ページからキーワードに一致するレースのrace_idを特定（±1日シフト対応）"""
    # 0. 手動フォールバック辞書をチェック
    if label and label in MANUAL_RACE_IDS:
        return MANUAL_RACE_IDS[label]

    # 1. 当日 → 前日 → 翌日 の順に検索
    for day_offset in [0, -1, 1]:
        try:
            dt = datetime.strptime(date, "%Y%m%d") + timedelta(days=day_offset)
        except ValueError:
            continue
        shifted = dt.strftime("%Y%m%d")

        url = f"https://db.netkeiba.com/race/list/{shifted}/"
        content = safe_request(url)
        if content is None:
            continue

        soup = BeautifulSoup(content, "lxml")
        links = soup.select("a[href*='/race/']")

        for link in links:
            href = link.get("href", "")
            text = link.get_text(strip=True)
            match = re.search(r"/race/(\d{12})/", href)
            if match and keyword in text:
                if day_offset != 0:
                    print(f"  📅 日付シフト: {date} → {shifted} で発見")
                return match.group(1)

    return None


def _parse_race_info(soup: BeautifulSoup) -> tuple[str, str, str, str]:
    """レースページからレース名・距離・surface・馬場状態を抽出"""
    # レース名
    race_name_tag = soup.select_one("h1")
    race_name = race_name_tag.get_text(strip=True) if race_name_tag else "不明"

    # レース情報（複数セレクタ対応）
    race_data = soup.select_one(
        ".racedata, .data_intro, .RaceData01, .Race_Data01"
    )
    info_text = race_data.get_text() if race_data else ""

    distance = ""
    surface = ""
    going = ""
    dist_match = re.search(r"(芝|ダ)(\d+)m", info_text)
    if dist_match:
        surface = "芝" if dist_match.group(1) == "芝" else "ダート"
        distance = dist_match.group(2)
    going_match = re.search(r"(良|稍重|重|不良)", info_text)
    if going_match:
        going = going_match.group(1)

    return race_name, distance, surface, going


def _scrape_from_race_netkeiba(race_id: str) -> pd.DataFrame | None:
    """race.netkeiba.com から結果取得（上がり3F・通過順あり）"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    content = safe_request(url)
    if content is None:
        return None

    soup = BeautifulSoup(content, "lxml")

    # race.netkeiba.comのテーブルセレクタ（複数候補）
    table = soup.select_one(
        "table.RaceTable01, table.race_table_01, "
        "table.Shutuba_Table, table.ResultTableWrap"
    )
    if table is None:
        # tableタグをすべて走査して、ヘッダ行に「着順」を含むものを探す
        for t in soup.select("table"):
            header_text = t.get_text()[:200]
            if "着順" in header_text and "馬名" in header_text:
                table = t
                break

    if table is None:
        return None

    try:
        df = pd.read_html(StringIO(str(table)), header=0)[0]
    except Exception:
        return None

    # 上がり3Fまたは通過順カラムが含まれているか確認
    col_text = " ".join(str(c) for c in df.columns)
    has_last3f = any(k in col_text for k in ["上り", "上がり", "ﾗｽﾄ"])
    has_passing = any(k in col_text for k in ["通過", "ｺｰﾅｰ"])

    if not has_last3f and not has_passing:
        return None  # DB版と同等 → フォールバックへ

    race_name, distance, surface, going = _parse_race_info(soup)

    df["race_id"] = race_id
    df["race_name"] = race_name
    df["distance"] = distance
    df["surface"] = surface
    df["going"] = going
    df["year"] = race_id[:4]

    return df


def _scrape_from_db_netkeiba(race_id: str) -> pd.DataFrame | None:
    """DB版ページからレース結果テーブルを取得（フォールバック）"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    content = safe_request(url)
    if content is None:
        return None

    soup = BeautifulSoup(content, "lxml")
    race_name, distance, surface, going = _parse_race_info(soup)

    table = soup.select_one("table.race_table_01")
    if table is None:
        print(f"  ⚠️ テーブル未検出: {race_id}")
        return None

    try:
        df = pd.read_html(StringIO(str(table)), header=0)[0]
    except Exception as e:
        print(f"  ⚠️ テーブル解析失敗: {e}")
        return None

    df["race_id"] = race_id
    df["race_name"] = race_name
    df["distance"] = distance
    df["surface"] = surface
    df["going"] = going
    df["year"] = race_id[:4]

    return df


def scrape_race_result(race_id: str) -> pd.DataFrame | None:
    """レース結果テーブルを取得（race.netkeiba.com優先、DB版フォールバック）"""
    # まず race.netkeiba.com を試行（上がり3F・通過順あり）
    df = _scrape_from_race_netkeiba(race_id)
    if df is not None:
        return df

    # フォールバック: DB版
    return _scrape_from_db_netkeiba(race_id)


def clean_result_df(df: pd.DataFrame) -> pd.DataFrame:
    """カラム名の正規化"""
    rename_map = {}
    for col in df.columns:
        s = str(col).strip().replace("\u3000", "")
        if "着順" in s or s == "着順":
            rename_map[col] = "着順"
        elif s == "枠" or ("枠" in s and "番" in s):
            rename_map[col] = "枠番"
        elif s == "馬番":
            rename_map[col] = "馬番"
        elif "馬名" in s:
            rename_map[col] = "馬名"
        elif "性齢" in s:
            rename_map[col] = "性齢"
        elif "斤量" in s:
            rename_map[col] = "斤量"
        elif "騎手" in s:
            rename_map[col] = "騎手"
        elif s == "タイム":
            rename_map[col] = "タイム"
        elif "着差" in s:
            rename_map[col] = "着差"
        elif "通過" in s:
            rename_map[col] = "通過順"
        elif "上り" in s or "上がり" in s or "後3F" in s or s == "後3F":
            rename_map[col] = "上がり3F"
        elif "単勝" in s:
            rename_map[col] = "単勝オッズ"
        elif "人気" in s:
            rename_map[col] = "人気"
        elif "馬体重" in s:
            rename_map[col] = "馬体重"
        elif "調教師" in s or "厩舎" in s:
            rename_map[col] = "調教師"
        elif "賞金" in s:
            rename_map[col] = "賞金"
    if rename_map:
        df = df.rename(columns=rename_map)
    # 残ったカラム名の空白除去
    df.columns = [str(c).replace(" ", "").replace("\u3000", "") for c in df.columns]
    return df


def main() -> None:
    print("=" * 50)
    print("SAKURA ORACLE - データ収集開始")
    print(f"対象: {len(TARGET_RACES)}レース")
    print("=" * 50)

    all_results: list[pd.DataFrame] = []
    race_info_rows: list[dict] = []
    failed: list[str] = []

    pbar = tqdm(TARGET_RACES, desc="取得中")

    for race in pbar:
        label = race["label"]
        pbar.set_description(label)

        # Phase 1: race_id 特定
        race_id = find_race_id_from_date(race["date"], race["keyword"], label=label)
        if race_id is None:
            print(f"  ⚠️ {label}: race_id未特定 (日付: {race['date']})")
            failed.append(label)
            continue

        print(f"  ✅ {label} -> {race_id}")

        # Phase 2: 結果取得
        df = scrape_race_result(race_id)
        if df is not None and not df.empty:
            df = clean_result_df(df)
            df["label"] = label
            all_results.append(df)
            race_info_rows.append({
                "race_id": race_id,
                "label": label,
                "date": race["date"],
                "distance": df["distance"].iloc[0] if "distance" in df.columns else "",
                "surface": df["surface"].iloc[0] if "surface" in df.columns else "",
                "going": df["going"].iloc[0] if "going" in df.columns else "",
                "num_horses": len(df),
            })
            print(f"    -> {len(df)}頭取得")
        else:
            failed.append(label)

    pbar.close()

    # 保存
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(RAW_DIR / "race_results.csv", index=False, encoding="utf-8-sig")
        print(f"\n✅ race_results.csv: {len(combined)}行")

        info_df = pd.DataFrame(race_info_rows)
        info_df.to_csv(RAW_DIR / "race_info.csv", index=False, encoding="utf-8-sig")
        print(f"✅ race_info.csv: {len(info_df)}レース")

    if failed:
        print(f"\n⚠️ 取得失敗: {len(failed)}件")
        for f in failed:
            print(f"   - {f}")

    print(f"\n合計: {len(all_results)}レース取得完了")
    print("データ収集終了")


if __name__ == "__main__":
    main()
