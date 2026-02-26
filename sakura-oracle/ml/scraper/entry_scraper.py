"""
SAKURA ORACLE — 出馬表スクレイパー

netkeibaの出馬表ページ（shutuba.html）から出走馬情報を取得する。

使い方:
    from ml.scraper.entry_scraper import scrape_entries
    df = scrape_entries("202509040811")
"""

import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.race_scraper import safe_request


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

    # 馬番が存在し数値化可能なら使用、なければ連番を振る（枠順未確定時）
    if "馬番" in df.columns:
        df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
        if df["馬番"].notna().sum() > 0:
            df = df.dropna(subset=["馬番"]).copy()
            df["馬番"] = df["馬番"].astype(int)
        else:
            # 馬番が全てNaN（枠順未確定）→ 連番を振る
            print("  ⚠️ 馬番未確定 — 仮番号を割り当て")
            df = df[df["馬名"].notna()].copy()
            df["馬番"] = range(1, len(df) + 1)
    else:
        # 馬番カラム自体がない → 連番を振る
        print("  ⚠️ 馬番カラムなし — 仮番号を割り当て")
        df = df[df["馬名"].notna()].copy()
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
