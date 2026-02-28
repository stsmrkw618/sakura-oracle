"""
SAKURA ORACLE — 出馬表スクレイパー

netkeibaの出馬表ページ（shutuba.html）から出走馬情報を取得する。

使い方:
    from ml.scraper.entry_scraper import scrape_entries
    df = scrape_entries("202509040811")
"""

import json
import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.race_scraper import safe_request

# プロジェクトルート
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _scrape_waku_from_odds(race_id: str) -> dict[str, dict[str, int]] | None:
    """オッズページから枠番・馬番マッピングを取得する。

    shutubaページでJS描画のため枠番/馬番が取得できない場合のフォールバック。
    1. ローカルキャッシュ（data/odds_cache/{race_id}.json）を優先
    2. なければオッズページの静的HTMLから取得

    Returns:
        {馬名: {"枠番": int, "馬番": int, "単勝オッズ"?: float}} の辞書。失敗時はNone。
    """
    # ローカルキャッシュを優先（JS描画でWeb取得できないオッズもここに保存できる）
    cache_path = _PROJECT_ROOT / "data" / "odds_cache" / f"{race_id}.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached:
            print(f"  ✅ ローカルキャッシュからオッズ読み込み: {cache_path.name}")
            return cached

    url = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b1"
    content = safe_request(url)
    if content is None:
        return None

    soup = BeautifulSoup(content, "lxml")
    odds_div = soup.find("div", id="odds_view_form")
    if odds_div is None:
        return None

    table = odds_div.find("table")
    if table is None:
        return None

    # 手動パース（pd.read_htmlはカラム名の空白問題があるため）
    result: dict[str, dict[str, int]] = {}
    odds_map: dict[str, float] = {}
    for row in table.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 5:
            continue

        # td[0]=枠番, td[1]=馬番, td[4]=馬名, td[5]=オッズ
        waku_text = tds[0].get_text(strip=True)
        umaban_text = tds[1].get_text(strip=True)
        name_text = re.sub(r"\s+", "", tds[4].get_text(strip=True))

        waku = int(waku_text) if waku_text.isdigit() else 0
        umaban = int(umaban_text) if umaban_text.isdigit() else 0

        if name_text and umaban > 0:
            result[name_text] = {"枠番": waku, "馬番": umaban}

        # オッズも取得（利用可能な場合）
        if len(tds) > 5:
            odds_text = tds[5].get_text(strip=True)
            odds_val = pd.to_numeric(odds_text, errors="coerce")
            if pd.notna(odds_val) and odds_val > 0:
                odds_map[name_text] = float(odds_val)

    if odds_map:
        # オッズが取得できた場合、resultに追加
        for name, odds in odds_map.items():
            if name in result:
                result[name]["単勝オッズ"] = odds

    return result if result else None


def scrape_entries(race_id: str) -> pd.DataFrame:
    """出馬表ページから出走馬情報を取得する。

    Args:
        race_id: netkeibaのrace_id（12桁）

    Returns:
        DataFrame with columns:
            枠番, 馬番, 馬名, 性齢, 斤量, 騎手, 単勝オッズ, 人気
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    content = safe_request(url)
    if content is None:
        raise RuntimeError(f"出馬表取得失敗: {url}")

    soup = BeautifulSoup(content, "lxml")

    # テーブル検出（複数セレクタ対応）
    table = soup.select_one(
        "table.Shutuba_Table, table.RaceTable01, table.race_table_01"
    )
    if table is None:
        # フォールバック: ヘッダに「馬番」を含むテーブルを検索
        for t in soup.select("table"):
            header = t.get_text()[:300]
            if "馬番" in header and "馬名" in header:
                table = t
                break

    if table is None:
        raise RuntimeError(f"出馬表テーブル未検出: {url}")

    try:
        df = pd.read_html(StringIO(str(table)), header=0)[0]
    except Exception as e:
        raise RuntimeError(f"出馬表パース失敗: {e}") from e

    # カラム名正規化（スペース・全角を除去してマッチ）
    rename_map: dict[str, str] = {}
    for col in df.columns:
        s = str(col).strip().replace("\u3000", "").replace(" ", "")
        if s == "枠" or s == "枠番" or ("枠" in s and "番" in s):
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
        elif "単勝" in s or "オッズ" in s:
            rename_map[col] = "単勝オッズ"
        elif s == "人気":
            rename_map[col] = "人気"
        elif "馬体重" in s:
            rename_map[col] = "馬体重"

    if rename_map:
        df = df.rename(columns=rename_map)

    # ヘッダ重複行を除外（先頭行がカラム名と同じ文字列のケース）
    if "馬名" in df.columns:
        df = df[df["馬名"] != "馬名"].copy()

    # 馬番が存在し数値化可能なら使用、なければオッズページから取得
    umaban_resolved = False
    waku_map: dict[str, dict] | None = None
    if "馬番" in df.columns:
        df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
        if df["馬番"].notna().sum() > 0:
            df = df.dropna(subset=["馬番"]).copy()
            df["馬番"] = df["馬番"].astype(int)
            umaban_resolved = True

    if not umaban_resolved:
        # shutubaページで馬番が取れない場合、オッズページから取得
        df = df[df["馬名"].notna()].copy()
        waku_map = _scrape_waku_from_odds(race_id)
        if waku_map:
            # 馬名のクリーニング（マッチ用に先に実施）
            df["馬名"] = df["馬名"].apply(
                lambda x: re.sub(r"\s+", "", str(x).strip()) if pd.notna(x) else x
            )
            matched = 0
            for idx, row in df.iterrows():
                name = row["馬名"]
                if name in waku_map:
                    df.at[idx, "馬番"] = waku_map[name]["馬番"]
                    df.at[idx, "枠番"] = waku_map[name]["枠番"]
                    matched += 1
            if matched > 0:
                df["馬番"] = df["馬番"].astype(int)
                df["枠番"] = df["枠番"].astype(int)
                print(f"  ✅ オッズページから枠番・馬番取得: {matched}/{len(df)}頭")
                umaban_resolved = True
            else:
                print("  ⚠️ オッズページの馬名マッチ失敗 — 仮番号を割り当て")
                df["馬番"] = range(1, len(df) + 1)
        else:
            print("  ⚠️ 馬番未確定 — 仮番号を割り当て")
            df["馬番"] = range(1, len(df) + 1)

    # 枠番を数値化（未確定なら0）
    if "枠番" in df.columns:
        df["枠番"] = pd.to_numeric(df["枠番"], errors="coerce").fillna(0).astype(int)
    else:
        df["枠番"] = 0

    # 馬名のクリーニング（改行・空白除去）
    if "馬名" in df.columns:
        df["馬名"] = df["馬名"].apply(
            lambda x: re.sub(r"\s+", "", str(x).strip()) if pd.notna(x) else x
        )

    # horse_id 抽出（馬名リンクから /horse/{horse_id}/ を取得）
    horse_id_map: dict[str, str] = {}
    for link in soup.select('a[href*="/horse/"]'):
        href = link.get("href", "")
        text = re.sub(r"\s+", "", link.get_text(strip=True))
        m = re.search(r"/horse/(\d{9,10})", href)
        if m and text and len(text) > 1:
            horse_id_map[text] = m.group(1)

    if "馬名" in df.columns:
        df["horse_id"] = df["馬名"].map(horse_id_map).fillna("")
        matched = (df["horse_id"] != "").sum()
        print(f"  horse_id マッチ: {matched}/{len(df)}頭")

    # オッズ・人気を数値化
    if "単勝オッズ" in df.columns:
        df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")
    if "人気" in df.columns:
        df["人気"] = pd.to_numeric(df["人気"], errors="coerce")

    # shutubaページでオッズが取れない場合、オッズページから補完
    odds_missing = (
        "単勝オッズ" not in df.columns
        or df["単勝オッズ"].notna().sum() == 0
    )
    if odds_missing:
        # オッズページを取得（枠番取得時に既にwaku_mapがあれば再利用）
        if waku_map is None:
            waku_map = _scrape_waku_from_odds(race_id)
        if waku_map:
            odds_count = 0
            for idx, row in df.iterrows():
                name = row.get("馬名", "")
                if name in waku_map and "単勝オッズ" in waku_map[name]:
                    df.at[idx, "単勝オッズ"] = waku_map[name]["単勝オッズ"]
                    odds_count += 1
            if odds_count > 0:
                df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")
                print(f"  ✅ オッズページからオッズ取得: {odds_count}/{len(df)}頭")

    # 斤量を数値化
    if "斤量" in df.columns:
        df["斤量"] = pd.to_numeric(df["斤量"], errors="coerce")

    print(f"  出馬表取得: {len(df)}頭 (race_id={race_id})")
    return df


if __name__ == "__main__":
    # テスト用: 引数にrace_idを指定
    if len(sys.argv) < 2:
        print("使い方: py ml/scraper/entry_scraper.py <race_id>")
        sys.exit(1)
    rid = sys.argv[1]
    result = scrape_entries(rid)
    print(result[["枠番", "馬番", "馬名", "騎手"]].to_string(index=False))
