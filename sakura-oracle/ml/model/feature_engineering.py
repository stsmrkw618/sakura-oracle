"""
SAKURA ORACLE — 特徴量エンジニアリング

使い方:
    PYTHONIOENCODING=utf-8 py ml/model/feature_engineering.py

入力: data/raw/race_results.csv
出力: data/features.csv
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.config import DATA_DIR, RAW_DIR

# === レースメタデータ（label → 距離・コース） ===
RACE_META: dict[str, dict[str, str | int]] = {
    "桜花賞": {"distance": 1600, "surface": "芝", "venue": "阪神", "grade": 5},
    "チューリップ賞": {"distance": 1600, "surface": "芝", "venue": "阪神", "grade": 3},
    "フィリーズレビュー": {"distance": 1400, "surface": "芝", "venue": "阪神", "grade": 3},
    "クイーンC": {"distance": 1600, "surface": "芝", "venue": "東京", "grade": 3},
    "フェアリーS": {"distance": 1600, "surface": "芝", "venue": "中山", "grade": 3},
    "阪神JF": {"distance": 1600, "surface": "芝", "venue": "阪神", "grade": 5},
    "アルテミスS": {"distance": 1600, "surface": "芝", "venue": "東京", "grade": 3},
    "ファンタジーS": {"distance": 1400, "surface": "芝", "venue": "京都", "grade": 3},
    "オークス": {"distance": 2400, "surface": "芝", "venue": "東京", "grade": 5},
    "秋華賞": {"distance": 2000, "surface": "芝", "venue": "京都", "grade": 5},
    "ローズS": {"distance": 1800, "surface": "芝", "venue": "阪神", "grade": 3},
    "紫苑S": {"distance": 2000, "surface": "芝", "venue": "中山", "grade": 3},
}


def get_race_base(label: str) -> str:
    """ラベルからレース基本名を取得（年度や世代部分を除去）"""
    for key in RACE_META:
        if key in label:
            return key
    return "不明"


def parse_time_to_seconds(time_str: str) -> float | None:
    """タイム文字列(M:SS.S)を秒数に変換"""
    if pd.isna(time_str):
        return None
    try:
        time_str = str(time_str).strip()
        match = re.match(r"(\d+):(\d+\.?\d*)", time_str)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds
    except (ValueError, AttributeError):
        pass
    return None


def parse_weight(weight_str: str) -> tuple[float | None, float | None]:
    """馬体重文字列 '480(+4)' → (480, 4)"""
    if pd.isna(weight_str):
        return None, None
    try:
        s = str(weight_str).strip()
        match = re.match(r"(\d+)\(([+-]?\d+)\)", s)
        if match:
            return float(match.group(1)), float(match.group(2))
        # 体重のみ
        match2 = re.match(r"(\d+)", s)
        if match2:
            return float(match2.group(1)), None
    except (ValueError, AttributeError):
        pass
    return None, None


def calc_speed_index(time_sec: float | None, distance: int, going: str) -> float | None:
    """
    独自スピード指数
    基準タイム - 実タイム をスケーリングし50を基準とする
    馬場差補正付き
    """
    if time_sec is None or pd.isna(time_sec) or distance == 0:
        return None

    # 距離別基準タイム（良馬場、G2-G3レベル）
    base_times: dict[int, float] = {
        1200: 69.5,
        1400: 82.0,
        1600: 94.5,
        1800: 107.0,
        2000: 119.5,
        2400: 144.5,
    }

    # 最も近い距離の基準タイムを取得
    closest = min(base_times.keys(), key=lambda d: abs(d - distance))
    base = base_times[closest]

    # 距離が異なる場合の補正（1ハロン=200mあたり12.5秒で補正）
    if closest != distance:
        base += (distance - closest) / 200 * 12.5

    # 馬場差補正
    going_adj: dict[str, float] = {"良": 0, "稍重": 0.5, "重": 1.0, "不良": 2.0}
    adj = going_adj.get(str(going), 0)

    raw = (base + adj - time_sec) * 10 + 50
    return round(raw, 1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """メイン特徴量生成"""
    print("特徴量生成開始...")

    # === メタデータ復元 ===
    df["race_base"] = df["label"].apply(get_race_base)
    df["distance_m"] = df["race_base"].map(lambda x: RACE_META.get(x, {}).get("distance", 0))
    df["surface_type"] = df["race_base"].map(lambda x: RACE_META.get(x, {}).get("surface", ""))
    df["venue"] = df["race_base"].map(lambda x: RACE_META.get(x, {}).get("venue", ""))
    df["grade_encoded"] = df["race_base"].map(lambda x: RACE_META.get(x, {}).get("grade", 1))

    # === 着順を数値化（除外・中止を除外） ===
    df["着順_num"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["着順_num"]).copy()
    df["着順_num"] = df["着順_num"].astype(int)

    # === タイム → 秒数 ===
    df["time_sec"] = df["タイム"].apply(parse_time_to_seconds)

    # === 馬体重パース ===
    weight_data = df["馬体重"].apply(parse_weight)
    df["weight"] = weight_data.apply(lambda x: x[0])
    df["weight_diff"] = weight_data.apply(lambda x: x[1])

    # === スピード指数 ===
    df["speed_index"] = df.apply(
        lambda row: calc_speed_index(row["time_sec"], row["distance_m"], row["going"]),
        axis=1
    )

    # === 上がり3F → 数値 ===
    df["last3f"] = pd.to_numeric(df.get("上がり3F"), errors="coerce")

    # === 単勝オッズ → 数値 ===
    df["odds"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")

    # === 人気 → 数値 ===
    df["popularity"] = pd.to_numeric(df["人気"], errors="coerce")

    # === 馬番・枠番 → 数値 ===
    df["horse_number"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["frame_number"] = pd.to_numeric(df["枠番"], errors="coerce")

    # === 着順に基づくターゲット変数 ===
    df["is_win"] = (df["着順_num"] == 1).astype(int)
    df["is_show"] = (df["着順_num"] <= 3).astype(int)

    # === 同一馬の過去走集計（馬名ベース） ===
    # 日付でソートしてから過去走の特徴量を生成
    df = df.sort_values(["馬名", "race_id"]).reset_index(drop=True)

    # 馬ごとの通算成績
    horse_stats = []
    for _, group in df.groupby("馬名"):
        g = group.copy()
        # 累積計算（その行を含めない＝過去走のみ）
        g["total_runs"] = range(len(g))
        g["total_wins"] = g["is_win"].cumsum().shift(1, fill_value=0)
        g["total_show"] = g["is_show"].cumsum().shift(1, fill_value=0)
        g["win_rate"] = np.where(g["total_runs"] > 0, g["total_wins"] / g["total_runs"], 0)
        g["show_rate"] = np.where(g["total_runs"] > 0, g["total_show"] / g["total_runs"], 0)

        # 直近走成績（shift で前走を取得）
        g["last1_finish"] = g["着順_num"].shift(1)
        g["last2_finish"] = g["着順_num"].shift(2)
        g["last3_finish"] = g["着順_num"].shift(3)
        g["last1_last3f"] = g["last3f"].shift(1)
        g["last2_last3f"] = g["last3f"].shift(2)
        g["last1_speed"] = g["speed_index"].shift(1)

        # 上がり3F統計（過去走）
        g["avg_last3f"] = g["last3f"].expanding().mean().shift(1)
        g["best_last3f"] = g["last3f"].expanding().min().shift(1)

        # 阪神実績
        g["is_hanshin"] = (g["venue"] == "阪神").astype(int)
        g["hanshin_runs"] = g["is_hanshin"].cumsum().shift(1, fill_value=0)

        # マイル(1400-1800m)実績
        g["is_mile"] = g["distance_m"].between(1400, 1800).astype(int)
        g["mile_wins"] = (g["is_win"] * g["is_mile"]).cumsum().shift(1, fill_value=0)
        g["mile_runs"] = g["is_mile"].cumsum().shift(1, fill_value=0)
        g["mile_win_rate"] = np.where(g["mile_runs"] > 0, g["mile_wins"] / g["mile_runs"], 0)

        horse_stats.append(g)

    df = pd.concat(horse_stats, ignore_index=True)

    # === 騎手の累積成績（リーケージなし: cumsum + shift） ===
    df = df.sort_values(["騎手", "race_id"]).reset_index(drop=True)
    jockey_stats = []
    for _, group in df.groupby("騎手"):
        g = group.copy()
        g["jockey_total_runs"] = range(len(g))
        g["jockey_total_wins"] = g["is_win"].cumsum().shift(1, fill_value=0)
        g["jockey_win_rate"] = np.where(
            g["jockey_total_runs"] > 0,
            g["jockey_total_wins"] / g["jockey_total_runs"],
            0,
        )
        # G1勝利数（grade_encoded==5 かつ is_win==1）
        g["_is_g1_win"] = ((g["grade_encoded"] == 5) & (g["is_win"] == 1)).astype(int)
        g["jockey_g1_wins"] = g["_is_g1_win"].cumsum().shift(1, fill_value=0)
        jockey_stats.append(g)
    df = pd.concat(jockey_stats, ignore_index=True)
    df = df.drop(columns=["_is_g1_win"], errors="ignore")

    # === 調教師の累積成績（新規追加） ===
    if "調教師" in df.columns:
        df = df.sort_values(["調教師", "race_id"]).reset_index(drop=True)
        trainer_stats = []
        for _, group in df.groupby("調教師"):
            g = group.copy()
            g["trainer_total_runs"] = range(len(g))
            g["trainer_total_wins"] = g["is_win"].cumsum().shift(1, fill_value=0)
            g["trainer_win_rate"] = np.where(
                g["trainer_total_runs"] > 0,
                g["trainer_total_wins"] / g["trainer_total_runs"],
                0,
            )
            trainer_stats.append(g)
        df = pd.concat(trainer_stats, ignore_index=True)

    # === 通過順特徴量（前走の位置取り） ===
    if "通過順" in df.columns:
        # 通過順は "3-3-3-2" のような形式。最初と最後を抽出
        def _parse_passing(val: str) -> tuple[float | None, float | None]:
            if pd.isna(val):
                return None, None
            try:
                parts = str(val).strip().split("-")
                first = float(parts[0])
                last = float(parts[-1])
                return first, last
            except (ValueError, IndexError):
                return None, None

        passing = df["通過順"].apply(_parse_passing)
        df["start_pos"] = passing.apply(lambda x: x[0])
        df["final_pos"] = passing.apply(lambda x: x[1])
        df["position_gain"] = df["start_pos"] - df["final_pos"]  # 正=追い込み

        # 脚質分類（通過順ベース）
        df["n_horses"] = df.groupby("race_id")["馬名"].transform("count")
        ratio = df["start_pos"] / df["n_horses"]
        df["running_style_code"] = pd.cut(
            ratio, bins=[-0.01, 1/8, 3/8, 5/8, 1.01],
            labels=[0, 1, 2, 3]
        ).astype(float)

        # 前走の通過順特徴量 + 脚質累積
        df = df.sort_values(["馬名", "race_id"]).reset_index(drop=True)
        last_pass = []
        for _, group in df.groupby("馬名"):
            g = group.copy()
            g["last1_start_pos"] = g["start_pos"].shift(1)
            g["last1_position_gain"] = g["position_gain"].shift(1)
            g["running_style_avg"] = g["running_style_code"].expanding().mean().shift(1)
            g["last1_running_style"] = g["running_style_code"].shift(1)
            last_pass.append(g)
        df = pd.concat(last_pass, ignore_index=True)
    else:
        df["last1_start_pos"] = np.nan
        df["last1_position_gain"] = np.nan

    # === 着差特徴量（前走の着差） ===
    if "着差" in df.columns:
        def _parse_margin(val: str) -> float | None:
            """着差文字列を数値化（馬身単位）"""
            if pd.isna(val):
                return 0.0  # 1着は着差なし = 0
            s = str(val).strip()
            if s in ("", "同着"):
                return 0.0
            # "クビ", "ハナ", "アタマ" などの変換
            margin_map = {"ハナ": 0.05, "アタマ": 0.1, "クビ": 0.25}
            for key, num in margin_map.items():
                if key in s:
                    return num
            # "1/2", "1.1/2", "2.1/2", "3/4" など
            try:
                s = s.replace("大差", "10")
                # "1.1/2" → 1.5, "2.1/2" → 2.5
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

        df["margin"] = df["着差"].apply(_parse_margin)
        df = df.sort_values(["馬名", "race_id"]).reset_index(drop=True)
        margin_stats = []
        for _, group in df.groupby("馬名"):
            g = group.copy()
            g["last1_margin"] = g["margin"].shift(1)
            margin_stats.append(g)
        df = pd.concat(margin_stats, ignore_index=True)
    else:
        df["last1_margin"] = np.nan

    # === フィールド強度（レース内の相対オッズ） ===
    # そのレース内のオッズの逆数合計に対する各馬の市場占有率
    df["_odds_inv"] = 1.0 / df["odds"].clip(lower=1.0)
    race_odds_sum = df.groupby("race_id")["_odds_inv"].transform("sum")
    df["field_strength"] = df["_odds_inv"] / race_odds_sum
    df = df.drop(columns=["_odds_inv"])

    # === ペース指標 ===
    df["front_half_time"] = df["time_sec"] - df["last3f"]
    df["pace_per_furlong"] = df["front_half_time"] / ((df["distance_m"] - 600) / 200)
    # 距離帯別z-score → 偏差値
    df["pace_deviation"] = df.groupby(
        pd.cut(df["distance_m"], bins=[0, 1500, 1700, 1900, 3000])
    )["pace_per_furlong"].transform(lambda x: (x - x.mean()) / x.std() * 10 + 50)

    # 前走ペース偏差値
    df = df.sort_values(["馬名", "race_id"]).reset_index(drop=True)
    pace_stats = []
    for _, group in df.groupby("馬名"):
        g = group.copy()
        g["last1_pace_deviation"] = g["pace_deviation"].shift(1)
        pace_stats.append(g)
    df = pd.concat(pace_stats, ignore_index=True)

    # === 推定逃げ・先行頭数（過去走平均ベース → リーケージなし） ===
    df["_is_front_est"] = (df["running_style_avg"] < 1.0).astype(float)
    df["n_front_runners_est"] = df.groupby("race_id")["_is_front_est"].transform("sum")
    df = df.drop(columns=["_is_front_est"])

    # === 出走間隔（rest_weeks）===
    # 同一馬の前回出走からの経過週数（重賞間のローテーション間隔）
    if "date" in df.columns:
        df["_race_date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        df = df.sort_values(["馬名", "race_id"]).reset_index(drop=True)
        rest_parts = []
        for _, group in df.groupby("馬名"):
            g = group.copy()
            g["rest_weeks"] = (g["_race_date"] - g["_race_date"].shift(1)).dt.days / 7.0
            rest_parts.append(g)
        df = pd.concat(rest_parts, ignore_index=True)
        df = df.drop(columns=["_race_date"])
        print(f"  rest_weeks計算完了: 中央値={df['rest_weeks'].median():.1f}週")
    else:
        df["rest_weeks"] = np.nan

    # === 血統データマージ ===
    pedigree_path = RAW_DIR / "horse_pedigree.csv"
    if pedigree_path.exists():
        print("  血統データ読み込み中...")
        ped_df = pd.read_csv(pedigree_path)
        # 馬名でマージ
        ped_cols = ["horse_name", "sire", "sire_category", "sire_category_code"]
        ped_merge = ped_df[[c for c in ped_cols if c in ped_df.columns]].copy()
        ped_merge = ped_merge.rename(columns={"horse_name": "馬名"})
        ped_merge = ped_merge.drop_duplicates(subset=["馬名"], keep="first")
        df = df.merge(ped_merge, on="馬名", how="left")
        df["sire_category_code"] = df["sire_category_code"].fillna(0).astype(int)

        # 父馬の累積成績（父馬の産駒勝率 — ベイズ平均で平滑化）
        if "sire" in df.columns:
            prior_win_rate = df["is_win"].mean()  # 全体平均（約6%）
            min_runs = 10  # 事前分布の重み
            df = df.sort_values(["sire", "race_id"]).reset_index(drop=True)
            sire_stats = []
            for _, group in df.groupby("sire"):
                g = group.copy()
                g["sire_runs"] = range(len(g))
                g["sire_wins"] = g["is_win"].cumsum().shift(1, fill_value=0)
                # ベイズ平均: (実績勝利 + 事前勝率 * min_runs) / (実績出走 + min_runs)
                g["sire_win_rate"] = (
                    (g["sire_wins"] + prior_win_rate * min_runs)
                    / (g["sire_runs"] + min_runs)
                )
                sire_stats.append(g)
            df = pd.concat(sire_stats, ignore_index=True)

        print(f"  血統マージ完了: sire_category分布 = {df['sire_category'].value_counts().to_dict()}")
    else:
        print("  ⚠️ horse_pedigree.csv 未検出 → 血統特徴量スキップ")
        df["sire_category_code"] = 0
        df["sire_win_rate"] = 0.0

    # === 出力カラム選択 ===
    feature_cols = [
        # ID
        "race_id", "label", "race_base", "馬名", "騎手",
        # 基本
        "horse_number", "frame_number", "weight", "weight_diff",
        "distance_m", "grade_encoded",
        # 成績
        "total_runs", "win_rate", "show_rate",
        # 近走
        "last1_finish", "last2_finish", "last3_finish",
        "last1_last3f", "last2_last3f", "last1_speed",
        # スピード
        "speed_index", "avg_last3f", "best_last3f",
        # コース適性
        "hanshin_runs", "mile_win_rate",
        # 騎手（リーケージ修正済み）
        "jockey_win_rate", "jockey_g1_wins",
        # 調教師（新規）
        "trainer_win_rate",
        # 通過順
        "last1_start_pos", "last1_position_gain",
        # 脚質（新規）
        "running_style_code", "running_style_avg", "last1_running_style",
        # 着差
        "last1_margin",
        # フィールド強度
        "field_strength",
        # ペース（新規）
        "pace_deviation", "last1_pace_deviation", "n_front_runners_est",
        # 出走間隔
        "rest_weeks",
        # 血統
        "sire_category_code", "sire_win_rate",
        # オッズ
        "odds", "popularity",
        # ターゲット
        "着順_num", "is_win", "is_show",
    ]

    out = df[[c for c in feature_cols if c in df.columns]].copy()

    # 欠損値補完
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if out[col].isnull().any():
            out[col] = out[col].fillna(out[col].median())

    return out


def main() -> None:
    print("=" * 50)
    print("SAKURA ORACLE - 特徴量エンジニアリング")
    print("=" * 50)

    csv_path = RAW_DIR / "race_results.csv"
    if not csv_path.exists():
        print(f"❌ {csv_path} が見つかりません。先にスクレイパーを実行してください。")
        return

    df = pd.read_csv(csv_path)
    print(f"入力: {len(df)}行")

    # race_info.csv から開催日をマージ（出走間隔計算用）
    race_info_path = RAW_DIR / "race_info.csv"
    if race_info_path.exists():
        race_info = pd.read_csv(race_info_path, dtype={"race_id": str, "date": str})
        df["race_id"] = df["race_id"].astype(str)
        df = df.merge(race_info[["race_id", "date"]], on="race_id", how="left")
        print(f"  race_info.csv マージ完了 (date列追加)")

    features = build_features(df)
    output_path = DATA_DIR / "features.csv"
    features.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ features.csv 保存: {len(features)}行 × {len(features.columns)}カラム")
    print(f"カラム: {list(features.columns)}")
    print(f"\n--- サンプル ---")
    print(features.head(3).to_string())


if __name__ == "__main__":
    main()
