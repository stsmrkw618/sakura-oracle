"""
SAKURA ORACLE — 馬別全戦績スクレイパー

netkeibaの馬ページから全戦績を取得し、feature_engineering.pyと同等の特徴量を構築する。
features.csvに過去データがない未知馬の特徴量補完に使用する。

使い方:
    from ml.scraper.horse_history_scraper import build_features_from_history
    features = build_features_from_history("2024105678", target_date="20260301")
"""

import re
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.scraper.race_scraper import safe_request
from ml.model.feature_engineering import (
    calc_speed_index,
    parse_time_to_seconds,
    parse_weight,
)


def _scrape_horse_history(horse_id: str) -> pd.DataFrame | None:
    """馬ページの戦績テーブルをスクレイプする。

    Args:
        horse_id: netkeibaの馬ID（9-10桁）

    Returns:
        戦績DataFrame or None（取得失敗時）
    """
    url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
    content = safe_request(url)
    if content is None:
        return None

    soup = BeautifulSoup(content, "lxml")

    # 戦績テーブルを検出
    table = soup.select_one("table.db_h_race_results")
    if table is None:
        # フォールバック: ヘッダに「日付」「着順」を含むテーブルを検索
        for t in soup.select("table"):
            header = t.get_text()[:300]
            if "日付" in header and "着順" in header:
                table = t
                break

    if table is None:
        return None

    try:
        df = pd.read_html(StringIO(str(table)), header=0)[0]
    except Exception:
        return None

    if df.empty:
        return None

    # カラム名正規化
    rename_map: dict[str, str] = {}
    for col in df.columns:
        s = str(col).strip().replace("\u3000", "").replace(" ", "")
        if s == "日付":
            rename_map[col] = "日付"
        elif s == "開催":
            rename_map[col] = "開催"
        elif s == "レース名":
            rename_map[col] = "レース名"
        elif s == "距離":
            rename_map[col] = "距離"
        elif s == "着順":
            rename_map[col] = "着順"
        elif s == "タイム":
            rename_map[col] = "タイム"
        elif "上り" in s or "上がり" in s or "後3F" in s:
            rename_map[col] = "上がり3F"
        elif "通過" in s:
            rename_map[col] = "通過順"
        elif "着差" in s:
            rename_map[col] = "着差"
        elif "馬体重" in s:
            rename_map[col] = "馬体重"
        elif s == "頭数":
            rename_map[col] = "頭数"
        elif s == "馬場":
            rename_map[col] = "馬場"

    if rename_map:
        df = df.rename(columns=rename_map)

    # 着順を数値化（除外・中止を除外）
    if "着順" in df.columns:
        df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
        df = df.dropna(subset=["着順"]).copy()
        df["着順"] = df["着順"].astype(int)

    return df


def _parse_distance(dist_str: str) -> tuple[str, int]:
    """距離文字列 '芝1600' → ('芝', 1600)、'ダ1200' → ('ダート', 1200)"""
    if pd.isna(dist_str):
        return "", 0
    s = str(dist_str).strip()
    m = re.match(r"(芝|ダ|障)(\d+)", s)
    if m:
        surface = "芝" if m.group(1) == "芝" else "ダート"
        return surface, int(m.group(2))
    return "", 0


def _parse_passing(val: str) -> float | None:
    """通過順文字列 '3-3-3-2' → 最初の数字（スタート位置）"""
    if pd.isna(val):
        return None
    try:
        parts = str(val).strip().split("-")
        return float(parts[0])
    except (ValueError, IndexError):
        return None


def _parse_margin(val: str) -> float | None:
    """着差文字列を数値化（馬身単位）"""
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if s in ("", "同着"):
        return 0.0
    margin_map = {"ハナ": 0.05, "アタマ": 0.1, "クビ": 0.25}
    for key, num in margin_map.items():
        if key in s:
            return num
    try:
        s = s.replace("大差", "10")
        # "1.1/2" → 1.5
        match = re.match(r"(\d+)\.(\d)/(\d)", s)
        if match:
            whole = int(match.group(1))
            num = int(match.group(2))
            den = int(match.group(3))
            return whole + num / den
        # "1/2", "3/4"
        match = re.match(r"(\d+)/(\d+)", s)
        if match:
            return int(match.group(1)) / int(match.group(2))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def _extract_venue(kaisu_str: str) -> str:
    """開催文字列 '1阪神' → '阪神'、'3東京' → '東京'"""
    if pd.isna(kaisu_str):
        return ""
    s = str(kaisu_str).strip()
    # 数字部分を除去
    venue = re.sub(r"\d+", "", s)
    return venue


def build_features_from_history(
    horse_id: str,
    target_date: str = "",
    horse_name: str = "",
) -> dict | None:
    """馬ページの全戦績から特徴量dictを構築する。

    Args:
        horse_id: netkeibaの馬ID
        target_date: 対象レース日 (YYYYMMDD形式) — 出走間隔計算用
        horse_name: 馬名（ログ出力用）

    Returns:
        特徴量dict（FEATURE_COLS_ALLに対応するキーを持つ）or None
    """
    display_name = horse_name or horse_id
    hist = _scrape_horse_history(horse_id)
    if hist is None or hist.empty:
        print(f"    ❌ {display_name}: 戦績取得失敗")
        return None

    # 日付でソート（古い順）
    if "日付" in hist.columns:
        hist["_date"] = pd.to_datetime(hist["日付"], errors="coerce")
        hist = hist.sort_values("_date").reset_index(drop=True)
    else:
        hist = hist.reset_index(drop=True)

    # 対象レース日より前のデータのみ使用
    if target_date and "_date" in hist.columns:
        target_dt = pd.to_datetime(target_date, format="%Y%m%d", errors="coerce")
        if target_dt is not None:
            hist = hist[hist["_date"] < target_dt].copy()
            if hist.empty:
                print(f"    ⚠️ {display_name}: 対象日以前のレース履歴なし")
                return None

    n_races = len(hist)
    print(f"    📊 {display_name}: {n_races}戦の履歴を取得")

    # === 距離・馬場パース ===
    if "距離" in hist.columns:
        parsed = hist["距離"].apply(_parse_distance)
        hist["_surface"] = parsed.apply(lambda x: x[0])
        hist["_distance"] = parsed.apply(lambda x: x[1])
    else:
        hist["_surface"] = ""
        hist["_distance"] = 0

    # === 開催場所 ===
    if "開催" in hist.columns:
        hist["_venue"] = hist["開催"].apply(_extract_venue)
    else:
        hist["_venue"] = ""

    # === 馬場状態（goingカラム）===
    going_col = "馬場" if "馬場" in hist.columns else None

    # === 着順（既に数値化済み）===
    finish = hist["着順"]

    # === タイム → 秒数 ===
    if "タイム" in hist.columns:
        hist["_time_sec"] = hist["タイム"].apply(parse_time_to_seconds)
    else:
        hist["_time_sec"] = np.nan

    # === スピード指数 ===
    def _calc_si(row: pd.Series) -> float | None:
        going = str(row.get(going_col, "良")) if going_col else "良"
        return calc_speed_index(row["_time_sec"], int(row["_distance"]), going)

    hist["_speed_index"] = hist.apply(_calc_si, axis=1)

    # === 上がり3F ===
    if "上がり3F" in hist.columns:
        hist["_last3f"] = pd.to_numeric(hist["上がり3F"], errors="coerce")
    else:
        hist["_last3f"] = np.nan

    # === 通過順 ===
    if "通過順" in hist.columns:
        hist["_start_pos"] = hist["通過順"].apply(_parse_passing)
    else:
        hist["_start_pos"] = np.nan

    # === 着差 ===
    if "着差" in hist.columns:
        hist["_margin"] = hist["着差"].apply(_parse_margin)
    else:
        hist["_margin"] = np.nan

    # === 馬体重 ===
    if "馬体重" in hist.columns:
        weight_data = hist["馬体重"].apply(parse_weight)
        hist["_weight"] = weight_data.apply(lambda x: x[0])
        hist["_weight_diff"] = weight_data.apply(lambda x: x[1])
    else:
        hist["_weight"] = np.nan
        hist["_weight_diff"] = np.nan

    # === 特徴量構築 ===
    features: dict = {}

    # total_runs: 過去走数
    features["total_runs"] = n_races

    # show_rate: 3着以内率
    show_count = (finish <= 3).sum()
    features["show_rate"] = show_count / n_races if n_races > 0 else 0

    # last1_finish: 直近の着順
    features["last1_finish"] = float(finish.iloc[-1])

    # speed_index: 直近のスピード指数
    valid_si = hist["_speed_index"].dropna()
    features["speed_index"] = float(valid_si.iloc[-1]) if len(valid_si) > 0 else np.nan

    # last1_speed: 1走前のスピード指数
    if len(valid_si) >= 1:
        features["last1_speed"] = float(valid_si.iloc[-1])
    else:
        features["last1_speed"] = np.nan

    # 上がり3F 統計
    valid_3f = hist["_last3f"].dropna()
    features["avg_last3f"] = float(valid_3f.mean()) if len(valid_3f) > 0 else np.nan
    features["best_last3f"] = float(valid_3f.min()) if len(valid_3f) > 0 else np.nan
    features["last1_last3f"] = float(valid_3f.iloc[-1]) if len(valid_3f) > 0 else np.nan
    features["last2_last3f"] = float(valid_3f.iloc[-2]) if len(valid_3f) >= 2 else np.nan

    # 阪神実績
    features["hanshin_runs"] = int((hist["_venue"] == "阪神").sum())

    # 直近の通過順（スタート位置）
    valid_sp = hist["_start_pos"].dropna()
    features["last1_start_pos"] = float(valid_sp.iloc[-1]) if len(valid_sp) > 0 else np.nan

    # 直近の着差
    valid_margin = hist["_margin"].dropna()
    features["last1_margin"] = float(valid_margin.iloc[-1]) if len(valid_margin) > 0 else np.nan

    # ペース偏差値
    # 前半タイム = タイム - 上がり3F、ハロン数 = (距離-600)/200
    if "上がり3F" in hist.columns and "タイム" in hist.columns:
        hist["_front_half"] = hist["_time_sec"] - hist["_last3f"]
        hist["_furlongs"] = (hist["_distance"] - 600) / 200
        hist["_pace_per_f"] = hist["_front_half"] / hist["_furlongs"].replace(0, np.nan)
        valid_pace = hist["_pace_per_f"].dropna()
        if len(valid_pace) > 1:
            # 偏差値化（平均=50、標準偏差=10）
            mean_p = valid_pace.mean()
            std_p = valid_pace.std()
            if std_p > 0:
                last_pace = float(valid_pace.iloc[-1])
                features["pace_deviation"] = (last_pace - mean_p) / std_p * 10 + 50
                if len(valid_pace) >= 2:
                    prev_pace = float(valid_pace.iloc[-2])
                    features["last1_pace_deviation"] = (prev_pace - mean_p) / std_p * 10 + 50
                else:
                    features["last1_pace_deviation"] = features["pace_deviation"]
            else:
                features["pace_deviation"] = 50.0
                features["last1_pace_deviation"] = 50.0
        elif len(valid_pace) == 1:
            features["pace_deviation"] = 50.0
            features["last1_pace_deviation"] = 50.0
        else:
            features["pace_deviation"] = np.nan
            features["last1_pace_deviation"] = np.nan
    else:
        features["pace_deviation"] = np.nan
        features["last1_pace_deviation"] = np.nan

    # 出走間隔（rest_weeks）: 対象レース日 - 直近レース日
    if target_date and "_date" in hist.columns:
        target_dt = pd.to_datetime(target_date, format="%Y%m%d", errors="coerce")
        last_date = hist["_date"].iloc[-1]
        if pd.notna(target_dt) and pd.notna(last_date):
            features["rest_weeks"] = (target_dt - last_date).days / 7.0

    # 馬体重
    valid_w = hist["_weight"].dropna()
    valid_wd = hist["_weight_diff"].dropna()
    features["weight"] = float(valid_w.iloc[-1]) if len(valid_w) > 0 else np.nan
    features["weight_diff"] = float(valid_wd.iloc[-1]) if len(valid_wd) > 0 else np.nan

    return features


if __name__ == "__main__":
    # テスト用: 引数にhorse_idを指定
    if len(sys.argv) < 2:
        print("使い方: py ml/scraper/horse_history_scraper.py <horse_id> [target_date]")
        print("例: py ml/scraper/horse_history_scraper.py 2022105678 20260301")
        sys.exit(1)

    hid = sys.argv[1]
    t_date = sys.argv[2] if len(sys.argv) > 2 else ""
    result = build_features_from_history(hid, target_date=t_date)
    if result:
        print("\n--- 構築された特徴量 ---")
        for k, v in sorted(result.items()):
            print(f"  {k}: {v}")
    else:
        print("特徴量構築失敗")
