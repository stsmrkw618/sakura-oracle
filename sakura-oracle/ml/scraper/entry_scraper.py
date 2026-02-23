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

    # カラム名正規化
    rename_map: dict[str, str] = {}
    for col in df.columns:
        s = str(col).strip().replace("\u3000", "")
        if s == "枠" or ("枠" in s and "番" in s):
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
        elif "人気" in s:
            rename_map[col] = "人気"
        elif "馬体重" in s:
            rename_map[col] = "馬体重"

    if rename_map:
        df = df.rename(columns=rename_map)

    # 馬番が数値でない行を除外（ヘッダ重複行など）
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    df = df.dropna(subset=["馬番"]).copy()
    df["馬番"] = df["馬番"].astype(int)

    # 枠番を数値化
    if "枠番" in df.columns:
        df["枠番"] = pd.to_numeric(df["枠番"], errors="coerce").fillna(0).astype(int)

    # 馬名のクリーニング（改行・空白除去）
    if "馬名" in df.columns:
        df["馬名"] = df["馬名"].apply(
            lambda x: re.sub(r"\s+", "", str(x).strip()) if pd.notna(x) else x
        )

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
