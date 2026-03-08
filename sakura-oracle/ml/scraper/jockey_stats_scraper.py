"""
SAKURA ORACLE — 騎手成績スクレイパー

netkeibaの騎手年度別成績ページから通算勝率・重賞勝利数を取得する。
features.csvに騎手データがない未知騎手の特徴量補完に使用する。

ページ構造:
    URL: https://db.netkeiba.com/jockey/result/{jockey_id}/
    テーブル: 年度別サマリー（年度, 順位, 1着, 2着, 3着, 着外, 重賞出走, 重賞勝利, ...）
    先頭行: ヘッダ重複行（年度="年度"）
    2行目: 累計行（年度="累計"）
    3行目以降: 各年度（新しい順）

使い方:
    from ml.scraper.jockey_stats_scraper import scrape_jockey_stats
    stats = scrape_jockey_stats("01088", target_date="20260301")
"""

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.race_scraper import safe_request


def scrape_jockey_stats(
    jockey_id: str,
    target_date: str = "",
    jockey_name: str = "",
) -> dict | None:
    """騎手ページから通算勝率・重賞勝利数を取得する。

    年度別サマリーテーブルをパースし、target_date以前の年度を集計する。
    jockey_g1_wins には重賞勝利数（G1/G2/G3合算）を使用する。

    Args:
        jockey_id: netkeibaの騎手ID（5桁）
        target_date: 対象レース日 (YYYYMMDD形式) — リーケージ防止用
        jockey_name: 騎手名（ログ出力用）

    Returns:
        {"jockey_win_rate": float, "jockey_g1_wins": int} or None
    """
    display_name = jockey_name or jockey_id

    # 騎手年度別成績ページ取得
    url = f"https://db.netkeiba.com/jockey/result/{jockey_id}/"
    content = safe_request(url)

    if content is None:
        print(f"    ❌ 騎手ページ取得失敗: {display_name} (ID={jockey_id})")
        return None

    soup = BeautifulSoup(content, "lxml")

    # テーブル検出
    table = soup.select_one("table")
    if table is None:
        print(f"    ❌ 騎手成績テーブル未検出: {display_name}")
        return None

    try:
        df = pd.read_html(StringIO(str(table)), header=0)[0]
    except Exception:
        print(f"    ❌ 騎手成績パース失敗: {display_name}")
        return None

    if df.empty:
        print(f"    ⚠️ 騎手成績なし: {display_name}")
        return None

    # カラム: 年度, 順位, 1着, 2着, 3着, 着外, 重賞, 重賞.1, ...
    # 重賞 = 重賞出走数, 重賞.1 = 重賞勝利数

    # ヘッダ重複行・累計行を除外し、年度行のみ抽出
    df = df[~df["年度"].isin(["年度", "累計"])].copy()
    df["_year"] = pd.to_numeric(df["年度"], errors="coerce")
    df = df.dropna(subset=["_year"]).copy()
    df["_year"] = df["_year"].astype(int)

    if df.empty:
        print(f"    ⚠️ 有効な年度データなし: {display_name}")
        return None

    # target_date以前の年度のみ使用（リーケージ防止）
    if target_date and len(target_date) >= 4:
        target_year = int(target_date[:4])
        # 対象年は除外（年度内の個別日付が不明なため保守的に）
        df = df[df["_year"] < target_year].copy()
        if df.empty:
            print(f"    ⚠️ {display_name}: {target_year}年以前の騎乗実績なし")
            return None

    # 数値カラムを変換
    for col in ["1着", "2着", "3着", "着外"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 重賞勝利数（重賞.1カラム）
    graded_col = "重賞.1" if "重賞.1" in df.columns else None
    if graded_col:
        df[graded_col] = pd.to_numeric(df[graded_col], errors="coerce").fillna(0).astype(int)

    # 集計
    total_wins = int(df["1着"].sum())
    total_rides = int(df[["1着", "2着", "3着", "着外"]].sum().sum())
    win_rate = total_wins / total_rides if total_rides > 0 else 0.0
    graded_wins = int(df[graded_col].sum()) if graded_col else 0

    print(
        f"    🏇 {display_name}: {total_rides}騎乗 {total_wins}勝 "
        f"勝率{win_rate:.3f} 重賞{graded_wins}勝"
    )

    return {
        "jockey_win_rate": win_rate,
        "jockey_g1_wins": graded_wins,
    }


if __name__ == "__main__":
    # テスト用: 引数にjockey_idを指定
    if len(sys.argv) < 2:
        print("使い方: py ml/scraper/jockey_stats_scraper.py <jockey_id> [target_date]")
        print("例: py ml/scraper/jockey_stats_scraper.py 01088 20260301")
        sys.exit(1)

    jid = sys.argv[1]
    t_date = sys.argv[2] if len(sys.argv) > 2 else ""
    result = scrape_jockey_stats(jid, target_date=t_date)
    if result:
        print("\n--- 騎手成績 ---")
        for k, v in sorted(result.items()):
            print(f"  {k}: {v}")
    else:
        print("騎手成績取得失敗")
