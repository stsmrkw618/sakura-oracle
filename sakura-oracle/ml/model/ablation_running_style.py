"""
SAKURA ORACLE — 脚質・ペース特徴量アブレーション実験

5つの新特徴量を1つずつ除外し、効いている特徴量を特定する。
比較対象:
  - baseline: 新特徴量なし（v8相当）
  - full: 全5特徴量あり（v9）
  - drop_X: Xを1つ除外（残り4つ）
  - only_X: Xだけ追加（残り0つ）

使い方:
    PYTHONIOENCODING=utf-8 py ml/model/ablation_running_style.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.config import DATA_DIR, BASE_DIR
from ml.model.backtest_all_races import (
    build_race_order,
    _get_available_features,
    _make_params_bin,
    _train_model,
    _load_payouts,
    _find_payout,
    BLEND_WEIGHT_A,
    BLEND_WEIGHT_B,
    GRADE_MAP,
)

# v8ベースライン特徴量（新特徴量なし）
_BASE_V8 = [
    "horse_number", "frame_number", "weight", "weight_diff",
    "distance_m", "grade_encoded",
    "total_runs", "show_rate",
    "last1_finish",
    "last1_last3f", "last2_last3f", "last1_speed",
    "speed_index", "avg_last3f", "best_last3f",
    "hanshin_runs",
    "jockey_win_rate", "jockey_g1_wins",
    "trainer_win_rate",
    "last1_start_pos",
    "last1_margin",
    "field_strength",
]

# 新特徴量5つ
NEW_FEATURES = [
    "running_style_avg",
    "last1_running_style",
    "pace_deviation",
    "last1_pace_deviation",
    "n_front_runners_est",
]


def run_backtest_with_features(
    df: pd.DataFrame,
    base_features: list[str],
    payouts: dict,
) -> list[dict]:
    """指定した特徴量セットでWalk-Forwardバックテストを実行"""
    feat_all = _get_available_features(df, base_features + ["odds", "popularity"])
    feat_no_odds = _get_available_features(
        df, [c for c in base_features if c != "field_strength"]
    )

    races = build_race_order(df)
    results = []
    min_train_size = 50

    for race in races:
        label = race["label"]
        test_df = df[df["label"] == label].copy()
        if test_df.empty:
            continue

        current_race_id = race["race_id"]
        train_df = df[df["race_id"].astype(str) < current_race_id].copy()
        if len(train_df) < min_train_size:
            continue

        params_win = _make_params_bin(scale_pos_weight=16.851)
        params_show = _make_params_bin(scale_pos_weight=5.020)

        X_train_all = train_df[feat_all].values
        X_test_all = test_df[feat_all].values
        X_train_no = train_df[feat_no_odds].values
        X_test_no = test_df[feat_no_odds].values

        y_win = train_df["is_win"].values
        y_show = train_df["is_show"].values

        model_a_win = _train_model(X_train_all, y_win, params_win)
        pred_a_win = model_a_win.predict_proba(X_test_all)[:, 1]
        model_a_show = _train_model(X_train_all, y_show, params_show)
        pred_a_show = model_a_show.predict_proba(X_test_all)[:, 1]

        model_b_win = _train_model(X_train_no, y_win, params_win)
        pred_b_win = model_b_win.predict_proba(X_test_no)[:, 1]
        model_b_show = _train_model(X_train_no, y_show, params_show)
        pred_b_show = model_b_show.predict_proba(X_test_no)[:, 1]

        pred_win = BLEND_WEIGHT_A * pred_a_win + BLEND_WEIGHT_B * pred_b_win
        pred_show = BLEND_WEIGHT_A * pred_a_show + BLEND_WEIGHT_B * pred_b_show

        test_df["pred_win"] = pred_win
        test_df["pred_show"] = pred_show

        # 評価
        top1 = test_df.nlargest(1, "pred_win").iloc[0]
        actual_winner = test_df[test_df["着順_num"] == 1]
        winner_name = actual_winner["馬名"].values[0] if len(actual_winner) > 0 else "?"
        win_hit = int(top1["馬名"] == winner_name)

        top3 = test_df.nlargest(3, "pred_show")
        actual_top3 = set(test_df[test_df["着順_num"] <= 3]["馬名"].values)
        pred_top3 = set(top3["馬名"].values)
        show_hits = len(actual_top3 & pred_top3)

        win_return = float(top1["odds"]) * 100 if top1["着順_num"] == 1 else 0.0

        show_return_total = 0.0
        for _, row in top3.iterrows():
            if row["着順_num"] <= 3:
                show_return_total += row["odds"] * 0.3 * 100

        # 組合せ馬券
        actual_top2_nums = set(test_df[test_df["着順_num"] <= 2]["horse_number"].values)
        actual_top3_nums = set(test_df[test_df["着順_num"] <= 3]["horse_number"].values)
        pred_top3_nums = set(test_df.nlargest(3, "pred_win")["horse_number"].values)
        pred_top5_nums = set(test_df.nlargest(5, "pred_win")["horse_number"].values)
        pred_top2_nums = set(test_df.nlargest(2, "pred_win")["horse_number"].values)

        quinella_box3_hit = int(len(pred_top3_nums & actual_top2_nums) >= 2)
        wide_top2_hit = int(len(pred_top2_nums & actual_top3_nums) >= 2)
        trio_box5_hit = int(len(pred_top5_nums & actual_top3_nums) >= 3)

        race_payouts = payouts.get(current_race_id, {})
        quinella_roi = 0.0
        wide_roi = 0.0
        trio5_roi = 0.0

        if race_payouts:
            if quinella_box3_hit:
                payout = _find_payout(race_payouts, "quinella", actual_top2_nums)
                quinella_roi = payout / 300.0
            if wide_top2_hit:
                payout = _find_payout(race_payouts, "wide", pred_top2_nums)
                wide_roi = payout / 100.0
            if trio_box5_hit:
                payout = _find_payout(race_payouts, "trio", actual_top3_nums)
                trio5_roi = payout / 1000.0

        results.append({
            "label": label,
            "year": race["year"],
            "grade": race["grade"],
            "win_hit": win_hit,
            "show_hits": show_hits,
            "win_return": win_return,
            "show_return": show_return_total,
            "quinella_box3_hit": quinella_box3_hit,
            "wide_top2_hit": wide_top2_hit,
            "trio_box5_hit": trio_box5_hit,
            "quinella_roi": quinella_roi,
            "wide_roi": wide_roi,
            "trio5_roi": trio5_roi,
        })

    return results


def summarize(results: list[dict]) -> dict:
    """結果を1行のメトリクスに集約"""
    n = len(results)
    if n == 0:
        return {}

    win_hits = sum(r["win_hit"] for r in results)
    show_hits = sum(r["show_hits"] for r in results)
    win_return = sum(r["win_return"] for r in results)
    show_return = sum(r["show_return"] for r in results)

    quinella_hits = sum(r["quinella_box3_hit"] for r in results)
    wide_hits = sum(r["wide_top2_hit"] for r in results)
    trio5_hits = sum(r["trio_box5_hit"] for r in results)

    quinella_return = sum(r["quinella_roi"] * 300 for r in results)
    wide_return = sum(r["wide_roi"] * 100 for r in results)
    trio5_return = sum(r["trio5_roi"] * 1000 for r in results)

    # ホールドアウト
    train_r = [r for r in results if r["year"] < 2025]
    test_r = [r for r in results if r["year"] >= 2025]
    train_roi = sum(r["win_return"] for r in train_r) / (len(train_r) * 100) if train_r else 0
    test_roi = sum(r["win_return"] for r in test_r) / (len(test_r) * 100) if test_r else 0
    deg = round(test_roi / train_roi, 2) if train_roi > 0 else 0

    return {
        "n": n,
        "win_hit": f"{win_hits}/{n} ({win_hits/n:.0%})",
        "show_hit": f"{show_hits}/{n*3} ({show_hits/(n*3):.0%})",
        "win_roi": f"{win_return/(n*100):.0%}",
        "show_roi": f"{show_return/(n*300):.0%}",
        "quinella": f"{quinella_hits}/{n} ({quinella_return/(n*300):.0%})",
        "wide": f"{wide_hits}/{n} ({wide_return/(n*100):.0%})",
        "trio5": f"{trio5_hits}/{n} ({trio5_return/(n*1000):.0%})",
        "holdout_deg": deg,
        # 数値（ソート用）
        "_win_roi": win_return / (n * 100),
        "_show_roi": show_return / (n * 300),
        "_quinella_roi": quinella_return / (n * 300),
        "_wide_roi": wide_return / (n * 100),
        "_trio5_roi": trio5_return / (n * 1000),
    }


def main() -> None:
    print("=" * 70)
    print("脚質・ペース特徴量 アブレーション実験")
    print("=" * 70)

    csv_path = DATA_DIR / "features.csv"
    df = pd.read_csv(csv_path)
    payouts = _load_payouts()
    print(f"入力: {len(df)}行, 配当: {len(payouts)}レース\n")

    # === 実験設定 ===
    configs = {}

    # baseline: 新特徴量なし
    configs["baseline (v8)"] = _BASE_V8

    # full: 全5特徴量
    configs["full (v9)"] = _BASE_V8 + NEW_FEATURES

    # drop_X: 1つ除外
    for feat in NEW_FEATURES:
        remaining = [f for f in NEW_FEATURES if f != feat]
        configs[f"drop_{feat}"] = _BASE_V8 + remaining

    # only_X: 1つだけ追加
    for feat in NEW_FEATURES:
        configs[f"only_{feat}"] = _BASE_V8 + [feat]

    # ペース3特徴量のみ（推奨候補）
    PACE_ONLY = ["pace_deviation", "last1_pace_deviation", "n_front_runners_est"]
    configs["pace_3_only"] = _BASE_V8 + PACE_ONLY

    # pace_deviation単体 vs ペース3 の比較用
    configs["pace_dev+n_front"] = _BASE_V8 + ["pace_deviation", "n_front_runners_est"]
    configs["pace_dev+last1_pace"] = _BASE_V8 + ["pace_deviation", "last1_pace_deviation"]

    # === 実行 ===
    all_results = {}
    total = len(configs)
    for i, (name, features) in enumerate(configs.items()):
        print(f"[{i+1}/{total}] {name} ({len(features)}特徴量)...", end=" ", flush=True)
        results = run_backtest_with_features(df, features, payouts)
        summary = summarize(results)
        all_results[name] = summary
        print(f"1着{summary['win_hit']}  単回{summary['win_roi']}  馬連{summary['quinella']}")

    # === 結果テーブル ===
    print(f"\n{'='*100}")
    print("アブレーション結果一覧")
    print(f"{'='*100}")
    header = f"{'構成':<30} {'1着的中':>10} {'単勝ROI':>8} {'複勝ROI':>8} {'馬連ROI':>10} {'ワイドROI':>10} {'三複5ROI':>10} {'劣化率':>6}"
    print(header)
    print("-" * 100)

    for name, s in all_results.items():
        print(
            f"{name:<30} {s['win_hit']:>10} {s['win_roi']:>8} {s['show_roi']:>8} "
            f"{s['quinella']:>10} {s['wide']:>10} {s['trio5']:>10} {s['holdout_deg']:>6}"
        )

    # === 特徴量ごとの貢献度（追加時の効果） ===
    print(f"\n{'='*70}")
    print("特徴量ごとの貢献度（only_X - baseline）")
    print(f"{'='*70}")
    base = all_results["baseline (v8)"]
    print(f"{'特徴量':<25} {'単勝ROI差':>10} {'複勝ROI差':>10} {'馬連ROI差':>10} {'三複5ROI差':>10}")
    print("-" * 70)
    for feat in NEW_FEATURES:
        s = all_results[f"only_{feat}"]
        d_win = s["_win_roi"] - base["_win_roi"]
        d_show = s["_show_roi"] - base["_show_roi"]
        d_q = s["_quinella_roi"] - base["_quinella_roi"]
        d_t5 = s["_trio5_roi"] - base["_trio5_roi"]
        print(f"{feat:<25} {d_win:>+10.0%} {d_show:>+10.0%} {d_q:>+10.0%} {d_t5:>+10.0%}")

    # === 推奨構成 ===
    print(f"\n{'='*70}")
    print("推奨構成の判断")
    print(f"{'='*70}")

    # 総合スコア: 各ROIの重み付き平均
    for name, s in all_results.items():
        score = (
            s["_win_roi"] * 0.2
            + s["_show_roi"] * 0.2
            + s["_quinella_roi"] * 0.2
            + s["_wide_roi"] * 0.2
            + s["_trio5_roi"] * 0.2
        )
        all_results[name]["_score"] = score

    ranked = sorted(all_results.items(), key=lambda x: -x[1]["_score"])
    print(f"\n総合スコア（5指標均等加重）:")
    for i, (name, s) in enumerate(ranked):
        marker = " <== BEST" if i == 0 else ""
        print(f"  {i+1}. {name:<30} score={s['_score']:.3f}{marker}")


if __name__ == "__main__":
    main()
