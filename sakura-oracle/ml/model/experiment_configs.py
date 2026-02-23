"""
SAKURA ORACLE — モデル構成最適化実験

6パターンのモデル構成をWalk-Forwardバックテストで比較し、
最も回収率が高い構成を特定する。

使い方:
    PYTHONIOENCODING=utf-8 py ml/model/experiment_configs.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.config import DATA_DIR
from ml.model.backtest_all_races import (
    build_race_order,
    _get_available_features,
    _make_params_bin,
    _train_model,
    GRADE_MAP,
)

# === 特徴量セット定義 ===
BASE_FEATURES = [
    "horse_number", "frame_number", "weight", "weight_diff",
    "distance_m", "grade_encoded",
    "total_runs", "win_rate", "show_rate",
    "last1_finish", "last2_finish", "last3_finish",
    "last1_last3f", "last2_last3f", "last1_speed",
    "speed_index", "avg_last3f", "best_last3f",
    "hanshin_runs", "mile_win_rate",
    "jockey_win_rate", "jockey_g1_wins",
    "trainer_win_rate",
    "last1_start_pos", "last1_position_gain",
    "last1_margin",
    "field_strength",
]

PEDIGREE_FEATURES = ["sire_category_code", "sire_win_rate"]
ODDS_FEATURES = ["odds", "popularity"]

# === 実験構成 ===
CONFIGS = {
    "A: 現状(A60+B40, 血統あり)": {
        "feat_all": BASE_FEATURES + PEDIGREE_FEATURES + ODDS_FEATURES,
        "feat_no_odds": BASE_FEATURES + PEDIGREE_FEATURES,
        "blend_a": 0.6, "blend_b": 0.4,
    },
    "B: ModelBのみ(血統あり)": {
        "feat_all": BASE_FEATURES + PEDIGREE_FEATURES,
        "feat_no_odds": BASE_FEATURES + PEDIGREE_FEATURES,
        "blend_a": 0.0, "blend_b": 1.0,
    },
    "C: ModelBのみ(血統なし)": {
        "feat_all": BASE_FEATURES,
        "feat_no_odds": BASE_FEATURES,
        "blend_a": 0.0, "blend_b": 1.0,
    },
    "D: A30+B70(血統あり)": {
        "feat_all": BASE_FEATURES + PEDIGREE_FEATURES + ODDS_FEATURES,
        "feat_no_odds": BASE_FEATURES + PEDIGREE_FEATURES,
        "blend_a": 0.3, "blend_b": 0.7,
    },
    "E: A30+B70(血統なし)": {
        "feat_all": BASE_FEATURES + ODDS_FEATURES,
        "feat_no_odds": BASE_FEATURES,
        "blend_a": 0.3, "blend_b": 0.7,
    },
    "F: A20+B80(血統なし)": {
        "feat_all": BASE_FEATURES + ODDS_FEATURES,
        "feat_no_odds": BASE_FEATURES,
        "blend_a": 0.2, "blend_b": 0.8,
    },
}


def run_experiment(df: pd.DataFrame, config: dict) -> dict:
    """1つの構成でWalk-Forwardバックテストを実行"""
    races = build_race_order(df)
    min_train_size = 50

    feat_all = _get_available_features(df, config["feat_all"])
    feat_no_odds = _get_available_features(df, config["feat_no_odds"])
    wa = config["blend_a"]
    wb = config["blend_b"]

    total_win_invest = 0
    total_win_return = 0.0
    total_show_invest = 0
    total_show_return = 0.0
    win_hits = 0
    show_hits = 0
    n_races = 0

    for race in races:
        label = race["label"]
        test_df = df[df["label"] == label].copy()
        if test_df.empty:
            continue

        current_race_id = race["race_id"]
        train_df = df[df["race_id"].astype(str) < current_race_id].copy()
        if len(train_df) < min_train_size:
            continue

        params_win = _make_params_bin(scale_pos_weight=8.128)
        params_show = _make_params_bin(scale_pos_weight=7.787)

        y_win = train_df["is_win"].values
        y_show = train_df["is_show"].values

        # Model A (with odds) — skip if blend_a == 0
        if wa > 0:
            X_train_a = train_df[feat_all].values
            X_test_a = test_df[feat_all].values
            model_a_win = _train_model(X_train_a, y_win, params_win)
            pred_a_win = model_a_win.predict_proba(X_test_a)[:, 1]
            model_a_show = _train_model(X_train_a, y_show, params_show)
            pred_a_show = model_a_show.predict_proba(X_test_a)[:, 1]
        else:
            pred_a_win = np.zeros(len(test_df))
            pred_a_show = np.zeros(len(test_df))

        # Model B (no odds)
        X_train_b = train_df[feat_no_odds].values
        X_test_b = test_df[feat_no_odds].values
        model_b_win = _train_model(X_train_b, y_win, params_win)
        pred_b_win = model_b_win.predict_proba(X_test_b)[:, 1]
        model_b_show = _train_model(X_train_b, y_show, params_show)
        pred_b_show = model_b_show.predict_proba(X_test_b)[:, 1]

        # Blend
        pred_win = wa * pred_a_win + wb * pred_b_win
        pred_show = wa * pred_a_show + wb * pred_b_show

        test_df["pred_win"] = pred_win
        test_df["pred_show"] = pred_show

        # Evaluate
        top1 = test_df.nlargest(1, "pred_win").iloc[0]
        actual_winner = test_df[test_df["着順_num"] == 1]
        winner_name = actual_winner["馬名"].values[0] if len(actual_winner) > 0 else "?"
        win_hit = int(top1["馬名"] == winner_name)
        win_hits += win_hit

        top3 = test_df.nlargest(3, "pred_show")
        actual_top3 = set(test_df[test_df["着順_num"] <= 3]["馬名"].values)
        pred_top3 = set(top3["馬名"].values)
        show_hits += len(actual_top3 & pred_top3)

        win_return = float(top1["odds"]) * 100 if top1["着順_num"] == 1 else 0.0
        show_return = 0.0
        for _, row in top3.iterrows():
            if row["着順_num"] <= 3:
                show_return += row["odds"] * 0.3 * 100

        total_win_invest += 100
        total_win_return += win_return
        total_show_invest += 300
        total_show_return += show_return
        n_races += 1

    win_roi = total_win_return / total_win_invest if total_win_invest > 0 else 0
    show_roi = total_show_return / total_show_invest if total_show_invest > 0 else 0

    return {
        "n_races": n_races,
        "win_hits": win_hits,
        "show_hits": show_hits,
        "win_roi": win_roi,
        "show_roi": show_roi,
        "win_profit": total_win_return - total_win_invest,
        "show_profit": total_show_return - total_show_invest,
    }


def main() -> None:
    print("=" * 70)
    print("SAKURA ORACLE - モデル構成最適化実験")
    print("=" * 70)

    csv_path = DATA_DIR / "features.csv"
    if not csv_path.exists():
        print("features.csv が見つかりません")
        return

    df = pd.read_csv(csv_path)
    print(f"データ: {len(df)}行\n")

    results = {}
    for name, config in CONFIGS.items():
        print(f"実験中: {name} ...", end="", flush=True)
        res = run_experiment(df, config)
        results[name] = res
        print(f" 単勝{res['win_roi']:.0%} / 複勝{res['show_roi']:.0%}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'構成':<30} {'単勝ROI':>8} {'複勝ROI':>8} {'1着的中':>8} {'複勝的中':>8} {'単勝利益':>10} {'複勝利益':>10}")
    print("-" * 70)

    best_name = ""
    best_combined = -999
    for name, res in results.items():
        n = res["n_races"]
        combined = res["win_roi"] * 0.5 + res["show_roi"] * 0.5  # 単複均等評価
        if combined > best_combined:
            best_combined = combined
            best_name = name
        print(
            f"  {name:<28} {res['win_roi']:>7.0%} {res['show_roi']:>7.0%} "
            f"{res['win_hits']/n:>7.1%} {res['show_hits']/(n*3):>7.1%} "
            f"{res['win_profit']:>+9.0f}円 {res['show_profit']:>+9.0f}円"
        )

    print(f"\n★ 最適構成: {best_name}")
    print(f"  単勝回収率: {results[best_name]['win_roi']:.0%}")
    print(f"  複勝回収率: {results[best_name]['show_roi']:.0%}")


if __name__ == "__main__":
    main()
