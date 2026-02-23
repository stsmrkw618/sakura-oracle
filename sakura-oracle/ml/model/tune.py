"""
SAKURA ORACLE — Optunaハイパーパラメータ最適化

Walk-Forward CVベースでLightGBMのパラメータを最適化する。
目的関数は実利益（単勝ROI）。

使い方:
    PYTHONIOENCODING=utf-8 py ml/model/tune.py

依存: optuna 4.7.0+
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.config import DATA_DIR, BASE_DIR
from ml.model.backtest_all_races import (
    build_race_order,
    _get_available_features,
    FEATURE_COLS_ALL,
    FEATURE_COLS_NO_ODDS,
    BLEND_WEIGHT_A,
    BLEND_WEIGHT_B,
)

# 最適化設定
N_TRIALS = 100
OPTIMIZE_TARGET = "win_roi"  # 単勝ROI最大化
MIN_TRAIN_SIZE = 50


def _train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
) -> lgb.LGBMClassifier:
    """早期停止付きLGBMClassifierを学習（backtest_all_races.pyと同じ）"""
    model = lgb.LGBMClassifier(**params)
    n = len(X_train)
    if n >= 100:
        split = int(n * 0.9)
        model.fit(
            X_train[:split], y_train[:split],
            eval_set=[(X_train[split:], y_train[split:])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def _walk_forward_eval(
    df: pd.DataFrame,
    params: dict,
    scale_pos_weight_win: float,
    scale_pos_weight_show: float,
    feat_all: list[str],
    feat_no_odds: list[str],
) -> dict[str, float]:
    """Walk-Forwardで全レースを評価し、ROI等を返す"""
    races = build_race_order(df)
    n_evaluated = 0
    total_win_return = 0.0
    total_show_return = 0.0
    win_hits = 0

    for race in races:
        label = race["label"]
        test_df = df[df["label"] == label].copy()
        if test_df.empty:
            continue

        current_race_id = race["race_id"]
        train_df = df[df["race_id"].astype(str) < current_race_id].copy()
        if len(train_df) < MIN_TRAIN_SIZE:
            continue

        params_win = {**params, "scale_pos_weight": scale_pos_weight_win}
        params_show = {**params, "scale_pos_weight": scale_pos_weight_show}

        X_train_all = train_df[feat_all].values
        X_test_all = test_df[feat_all].values
        X_train_no_odds = train_df[feat_no_odds].values
        X_test_no_odds = test_df[feat_no_odds].values

        y_win = train_df["is_win"].values
        y_show = train_df["is_show"].values

        try:
            # Model A（全特徴量 — 市場連動型）
            model_a_win = _train_model(X_train_all, y_win, params_win)
            pred_a_win = model_a_win.predict_proba(X_test_all)[:, 1]

            model_a_show = _train_model(X_train_all, y_show, params_show)
            pred_a_show = model_a_show.predict_proba(X_test_all)[:, 1]

            # Model B（オッズなし — エッジ検出型）
            model_b_win = _train_model(X_train_no_odds, y_win, params_win)
            pred_b_win = model_b_win.predict_proba(X_test_no_odds)[:, 1]

            model_b_show = _train_model(X_train_no_odds, y_show, params_show)
            pred_b_show = model_b_show.predict_proba(X_test_no_odds)[:, 1]
        except Exception:
            continue

        # ブレンド（A20:B80）
        pred_win = BLEND_WEIGHT_A * pred_a_win + BLEND_WEIGHT_B * pred_b_win
        pred_show = BLEND_WEIGHT_A * pred_a_show + BLEND_WEIGHT_B * pred_b_show

        test_df = test_df.copy()
        test_df["pred_win"] = pred_win
        test_df["pred_show"] = pred_show

        # 評価
        top1 = test_df.nlargest(1, "pred_win").iloc[0]
        actual_winner = test_df[test_df["着順_num"] == 1]
        winner_name = actual_winner["馬名"].values[0] if len(actual_winner) > 0 else "?"
        win_hit = int(top1["馬名"] == winner_name)
        win_hits += win_hit
        total_win_return += float(top1["odds"]) * 100 if top1["着順_num"] == 1 else 0.0

        top3 = test_df.nlargest(3, "pred_show")
        for _, row in top3.iterrows():
            if row["着順_num"] <= 3:
                total_show_return += row["odds"] * 0.3 * 100

        n_evaluated += 1

    if n_evaluated == 0:
        return {"win_roi": 0.0, "show_roi": 0.0, "n_races": 0}

    win_roi = total_win_return / (n_evaluated * 100)
    show_roi = total_show_return / (n_evaluated * 300)

    return {
        "win_roi": win_roi,
        "show_roi": show_roi,
        "win_hits": win_hits,
        "n_races": n_evaluated,
    }


def objective(trial: optuna.Trial, df: pd.DataFrame, feat_all: list[str], feat_no_odds: list[str]) -> float:
    """Optuna目的関数: Walk-Forward ROIを最大化"""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "random_state": 42,
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
    }

    # クラス不均衡対応の scale_pos_weight
    spw_win = trial.suggest_float("scale_pos_weight_win", 5.0, 20.0)
    spw_show = trial.suggest_float("scale_pos_weight_show", 1.5, 8.0)

    result = _walk_forward_eval(df, params, spw_win, spw_show, feat_all, feat_no_odds)

    # 中間報告
    trial.set_user_attr("win_roi", result["win_roi"])
    trial.set_user_attr("show_roi", result["show_roi"])
    trial.set_user_attr("n_races", result["n_races"])

    return result[OPTIMIZE_TARGET]


def main() -> None:
    print("=" * 60)
    print("SAKURA ORACLE - Optunaハイパーパラメータ最適化")
    print(f"目的: {OPTIMIZE_TARGET} 最大化")
    print(f"ブレンド: A={BLEND_WEIGHT_A} / B={BLEND_WEIGHT_B}")
    print(f"トライアル数: {N_TRIALS}")
    print("=" * 60)

    csv_path = DATA_DIR / "features.csv"
    if not csv_path.exists():
        print("❌ features.csv が見つかりません")
        return

    df = pd.read_csv(csv_path)
    print(f"入力: {len(df)}行 × {len(df.columns)}カラム\n")

    # 特徴量は固定（backtest_all_races.pyと同一）
    feat_all = _get_available_features(df, FEATURE_COLS_ALL)
    feat_no_odds = _get_available_features(df, FEATURE_COLS_NO_ODDS)

    print(f"特徴量（固定）: Model A {len(feat_all)}個 / Model B {len(feat_no_odds)}個\n")

    # まず現在のパラメータでベースラインを計測
    from ml.model.backtest_all_races import _make_params_bin
    print("--- ベースライン（現在のパラメータ） ---")
    current_params = _make_params_bin()
    # scale_pos_weightは別途渡すので除外
    base_params = {k: v for k, v in current_params.items() if k != "scale_pos_weight"}
    baseline = _walk_forward_eval(df, base_params, 8.128, 7.787, feat_all, feat_no_odds)
    print(f"  win_roi: {baseline['win_roi']:.3f}")
    print(f"  show_roi: {baseline['show_roi']:.3f}")
    print(f"  評価レース数: {baseline['n_races']}\n")

    # Optuna Study
    study = optuna.create_study(
        direction="maximize",
        study_name="sakura_oracle_tuning",
    )

    study.optimize(
        lambda trial: objective(trial, df, feat_all, feat_no_odds),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # 結果出力
    print("\n" + "=" * 60)
    print("最適化結果")
    print("=" * 60)

    best = study.best_trial
    print(f"\n最良トライアル #{best.number}")
    print(f"  {OPTIMIZE_TARGET}: {best.value:.3f}")
    print(f"  win_roi: {best.user_attrs.get('win_roi', '?'):.3f}")
    print(f"  show_roi: {best.user_attrs.get('show_roi', '?'):.3f}")
    print(f"  評価レース数: {best.user_attrs.get('n_races', '?')}")
    print(f"\nベースライン比較:")
    print(f"  現在 win_roi: {baseline['win_roi']:.3f} → 最適化 win_roi: {best.user_attrs.get('win_roi', 0):.3f}")
    print(f"  現在 show_roi: {baseline['show_roi']:.3f} → 最適化 show_roi: {best.user_attrs.get('show_roi', 0):.3f}")
    improvement = best.user_attrs.get("win_roi", 0) - baseline["win_roi"]
    print(f"  改善幅: {improvement:+.3f} ({improvement/max(baseline['win_roi'],0.001)*100:+.1f}%)")
    print(f"\n最適パラメータ:")
    for key, val in best.params.items():
        print(f"  {key}: {val}")

    # 結果をJSONに保存
    output = {
        "best_value": best.value,
        "optimize_target": OPTIMIZE_TARGET,
        "best_params": best.params,
        "best_attrs": {
            "win_roi": best.user_attrs.get("win_roi"),
            "show_roi": best.user_attrs.get("show_roi"),
            "n_races": best.user_attrs.get("n_races"),
        },
        "n_trials": N_TRIALS,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "win_roi": t.user_attrs.get("win_roi"),
                "show_roi": t.user_attrs.get("show_roi"),
            }
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ],
    }

    json_path = BASE_DIR / "ml" / "output" / "optuna_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ {json_path} に保存")


if __name__ == "__main__":
    main()
