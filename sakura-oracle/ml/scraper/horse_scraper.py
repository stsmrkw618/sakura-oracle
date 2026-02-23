"""
SAKURA ORACLE — 血統スクレイパー

使い方:
    PYTHONIOENCODING=utf-8 py ml/scraper/horse_scraper.py

処理フロー:
    1. キャッシュ済みレースHTMLから horse_id ↔ 馬名 マッピングを抽出
    2. 各馬の /horse/ped/{horse_id}/ ページから父・母・母父を取得
    3. 父系統カテゴリをマッピング
    4. data/raw/horse_pedigree.csv に出力
"""

import pickle
import re
import sys
import time
import random
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.config import (
    HEADERS, CACHE_DIR, RAW_DIR,
    REQUEST_TIMEOUT, MAX_RETRIES, MIN_WAIT, MAX_WAIT, BACKOFF_BASE,
)

# === 父系統マッピング ===
# netkeiba の「〇〇系」表記が取れない場合の手動マッピング
SIRE_LINE_MAP: dict[str, str] = {
    # サンデーサイレンス系
    "ディープインパクト": "サンデーサイレンス系",
    "ハーツクライ": "サンデーサイレンス系",
    "ステイゴールド": "サンデーサイレンス系",
    "ダイワメジャー": "サンデーサイレンス系",
    "ゴールドシップ": "サンデーサイレンス系",
    "オルフェーヴル": "サンデーサイレンス系",
    "キズナ": "サンデーサイレンス系",
    "エピファネイア": "サンデーサイレンス系",
    "サトノダイヤモンド": "サンデーサイレンス系",
    "リアルスティール": "サンデーサイレンス系",
    "シルバーステート": "サンデーサイレンス系",
    "サトノクラウン": "サンデーサイレンス系",
    "スワーヴリチャード": "サンデーサイレンス系",
    "サトノアラジン": "サンデーサイレンス系",
    "マカヒキ": "サンデーサイレンス系",
    "ワールドプレミア": "サンデーサイレンス系",
    "ジャスタウェイ": "サンデーサイレンス系",
    "レイデオロ": "サンデーサイレンス系",
    "サリオス": "サンデーサイレンス系",
    "コントレイル": "サンデーサイレンス系",
    "シャフリヤール": "サンデーサイレンス系",
    "イスラボニータ": "サンデーサイレンス系",
    "マイラプソディ": "サンデーサイレンス系",
    # キングカメハメハ系
    "キングカメハメハ": "キングカメハメハ系",
    "ロードカナロア": "キングカメハメハ系",
    "ルーラーシップ": "キングカメハメハ系",
    "ドゥラメンテ": "キングカメハメハ系",
    "レイパパレ": "キングカメハメハ系",
    "ホッコータルマエ": "キングカメハメハ系",
    # ミスタープロスペクター系（その他）
    "クロフネ": "ミスタープロスペクター系",
    "フレンチデピュティ": "ミスタープロスペクター系",
    "キンシャサノキセキ": "ミスタープロスペクター系",
    # ノーザンダンサー系
    "モーリス": "ノーザンダンサー系",
    "リオンディーズ": "ノーザンダンサー系",
    # ストームキャット系
    "ヘニーヒューズ": "ストームキャット系",
    # ロベルト系
    "スクリーンヒーロー": "ロベルト系",
    "モーリス": "ロベルト系",
    # その他
    "ドレフォン": "ストームキャット系",
    "マインドユアビスケッツ": "ミスタープロスペクター系",
    "サウスヴィグラス": "ミスタープロスペクター系",
    "ビッグアーサー": "サンデーサイレンス系",
    "サトノアーサー": "サンデーサイレンス系",
    "ブリックスアンドモルタル": "ストームキャット系",
    "ナダル": "ミスタープロスペクター系",
    "ニューイヤーズデイ": "ミスタープロスペクター系",
    # 追加: 「その他」から救出（約130頭）
    "キタサンブラック": "サンデーサイレンス系",
    "ミッキーアイル": "サンデーサイレンス系",
    "カレンブラックヒル": "サンデーサイレンス系",
    "ブラックタイド": "サンデーサイレンス系",
    "アグネスタキオン": "サンデーサイレンス系",
    "フジキセキ": "サンデーサイレンス系",
    "ダノンバラード": "サンデーサイレンス系",
    "リアルインパクト": "サンデーサイレンス系",
    "リーチザクラウン": "サンデーサイレンス系",
    "アドマイヤマーズ": "サンデーサイレンス系",  # ダイワメジャー産駒
    "サートゥルナーリア": "キングカメハメハ系",
    "ロジャーバローズ": "キングカメハメハ系",
    "デクラレーションオブウォー": "ノーザンダンサー系",
    "レッドファルクス": "ミスタープロスペクター系",
}

# netkeibaの系統表記 → トップレベル系統への二次マッピング
NETKEIBA_LINE_MAP: dict[str, str] = {
    "Halo系": "サンデーサイレンス系",
    "Danzig系": "ノーザンダンサー系",
    "Bird系": "ストームキャット系",
    "Prospector系": "ミスタープロスペクター系",
    "Roberto系": "ロベルト系",
    "Lyphard系": "ノーザンダンサー系",
    "Nureyev系": "ノーザンダンサー系",
    "Minister系": "ノーザンダンサー系",
    "Nijinsky系": "ノーザンダンサー系",
    "Storm Cat系": "ストームキャット系",
    "Storm Bird系": "ストームキャット系",
    "Mr. Prospector系": "ミスタープロスペクター系",
    "Deputy Minister系": "ノーザンダンサー系",
}

# 大分類マッピング（系統 → カテゴリコード）
SIRE_CATEGORY_ENCODE: dict[str, int] = {
    "サンデーサイレンス系": 1,
    "キングカメハメハ系": 2,
    "ミスタープロスペクター系": 3,
    "ノーザンダンサー系": 4,
    "ストームキャット系": 5,
    "ロベルト系": 6,
    "その他": 0,
}


def polite_sleep() -> None:
    """netkeiba用の礼儀正しい待機"""
    time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))


def safe_request(url: str, max_retries: int = MAX_RETRIES) -> bytes | None:
    """キャッシュ付き安全リクエスト（race_scraper.py と同じロジック）"""
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

        except Exception as e:
            print(f"  リトライ {attempt + 1}/{max_retries}: {e}")
            time.sleep(30)

    print(f"  ❌ 取得失敗: {url}")
    return None


def extract_horse_ids_from_cache() -> dict[str, str]:
    """キャッシュ済みレースHTMLから horse_id ↔ 馬名マッピングを抽出"""
    horse_map: dict[str, str] = {}

    race_pages = sorted(CACHE_DIR.glob("https___db_netkeiba_com_race_2*"))
    race_pages += sorted(CACHE_DIR.glob("https___race_netkeiba*"))

    for pkl_file in race_pages:
        try:
            with open(pkl_file, "rb") as f:
                content = pickle.load(f)
            soup = BeautifulSoup(content, "lxml")
            links = soup.select('a[href*="/horse/"]')
            for link in links:
                href = link.get("href", "")
                text = link.get_text(strip=True)
                m = re.search(r"/horse/(\d{9,10})", href)
                if m and text and len(text) > 1:
                    horse_map[m.group(1)] = text
        except Exception:
            continue

    return horse_map


def _extract_name(cell_text: str) -> str:
    """セルテキストから馬名を抽出（年号より前の部分、英語名除去）"""
    # "クロフネ1998 芦毛..." → "クロフネ"
    # "French Deputy(米)1992..." → "French Deputy"
    m = re.match(r"(.+?)[\d(（]", cell_text)
    name = m.group(1).strip() if m else cell_text.strip()
    # 英語名混在除去: "ハービンジャーHarbinger" → "ハービンジャー"
    cleaned = re.sub(r"[A-Za-z\s']+$", "", name).strip()
    return cleaned if cleaned else name


def _extract_sire_line(cell_text: str) -> str:
    """セルテキストから父系統名を抽出（「〇〇系」の部分）"""
    m = re.search(r"([^\s\[\]]+系)", cell_text)
    if m:
        return m.group(1)
    return ""


def scrape_pedigree(horse_id: str) -> dict | None:
    """馬の血統ページから父・母・母父を取得"""
    url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"
    content = safe_request(url)
    if content is None:
        return None

    soup = BeautifulSoup(content, "lxml")
    table = soup.select_one("table.blood_table")
    if table is None:
        return None

    rows = table.select("tr")
    if len(rows) < 17:
        return None

    try:
        # Row 0, Cell 0 (rowspan=16) = 父
        sire_cell = rows[0].select("td")[0]
        sire_text = sire_cell.get_text(strip=True)
        sire = _extract_name(sire_text)
        sire_line = _extract_sire_line(sire_text)

        # Row 16, Cell 0 (rowspan=16) = 母
        dam_cell = rows[16].select("td")[0]
        dam = _extract_name(dam_cell.get_text(strip=True))

        # Row 16, Cell 1 (rowspan=8) = 母父
        dam_cells = rows[16].select("td")
        bms = _extract_name(dam_cells[1].get_text(strip=True)) if len(dam_cells) > 1 else ""

        return {
            "sire": sire,
            "dam": dam,
            "bms": bms,
            "sire_line": sire_line,
        }
    except (IndexError, AttributeError):
        return None


def classify_sire_line(sire: str, sire_line_from_page: str) -> str:
    """父系統カテゴリを判定（手動マッピング優先）"""
    # 1. 手動マッピング最優先（キングカメハメハ系等のオーバーライド）
    if sire in SIRE_LINE_MAP:
        return SIRE_LINE_MAP[sire]

    # 2. netkeiba ページの系統表記
    if sire_line_from_page:
        # "Deputy Minister系" のような表記をトップレベルに変換
        for top_line in SIRE_CATEGORY_ENCODE:
            if top_line != "その他" and top_line in sire_line_from_page:
                return top_line
        # サンデーサイレンス系の亜系
        if "サンデーサイレンス" in sire_line_from_page:
            return "サンデーサイレンス系"
        # 3. NETKEIBA_LINE_MAP で二次マッピング
        for pattern, mapped_line in NETKEIBA_LINE_MAP.items():
            if pattern in sire_line_from_page:
                return mapped_line

    # 4. 不明
    return "その他"


def main() -> None:
    """メイン処理"""
    print("=" * 50)
    print("SAKURA ORACLE - 血統データ収集")
    print("=" * 50)

    # Step 1: キャッシュからhorse_idマッピング抽出
    print("\n[1/3] キャッシュHTMLから horse_id 抽出中...")
    horse_map = extract_horse_ids_from_cache()
    print(f"  → {len(horse_map)}頭のユニーク馬を検出")

    # 既存の horse_pedigree.csv があればスキップ対象を確認
    output_path = RAW_DIR / "horse_pedigree.csv"
    existing: set[str] = set()
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing = set(existing_df["horse_id"].astype(str))
        print(f"  → 既存データ: {len(existing)}頭（スキップ）")

    to_fetch = {hid: name for hid, name in horse_map.items() if hid not in existing}
    print(f"  → 新規取得対象: {len(to_fetch)}頭")

    # Step 2: 血統ページスクレイピング
    print(f"\n[2/3] 血統ページ取得中...")
    results: list[dict] = []

    # 既存データがあれば読み込み
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        results = existing_df.to_dict("records")

    pbar = tqdm(to_fetch.items(), desc="血統取得")
    failed = 0
    for horse_id, horse_name in pbar:
        pbar.set_description(f"{horse_name}")
        ped = scrape_pedigree(horse_id)
        if ped is None:
            failed += 1
            results.append({
                "horse_id": horse_id,
                "horse_name": horse_name,
                "sire": "不明",
                "dam": "不明",
                "bms": "不明",
                "sire_line": "",
                "sire_category": "その他",
                "sire_category_code": 0,
            })
            continue

        sire_category = classify_sire_line(ped["sire"], ped["sire_line"])
        results.append({
            "horse_id": horse_id,
            "horse_name": horse_name,
            "sire": ped["sire"],
            "dam": ped["dam"],
            "bms": ped["bms"],
            "sire_line": ped["sire_line"],
            "sire_category": sire_category,
            "sire_category_code": SIRE_CATEGORY_ENCODE.get(sire_category, 0),
        })

    pbar.close()

    # Step 3: CSV出力
    print(f"\n[3/3] CSV出力中...")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ horse_pedigree.csv: {len(df)}頭")
    print(f"   取得失敗: {failed}頭")

    # 系統別集計
    print(f"\n--- 父系統カテゴリ分布 ---")
    cat_counts = df["sire_category"].value_counts()
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count}頭")

    # 父馬別集計（上位10）
    print(f"\n--- 父馬 Top10 ---")
    sire_counts = df["sire"].value_counts().head(10)
    for sire, count in sire_counts.items():
        print(f"  {sire}: {count}頭")

    print("\n血統データ収集完了!")


if __name__ == "__main__":
    main()
