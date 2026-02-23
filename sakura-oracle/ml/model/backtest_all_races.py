"""
SAKURA ORACLE — 全重賞Walk-Forwardバックテスト

全レースに対してリーケージなしの時系列バックテストを実行。
各レースを予測する際、「そのレースより前のデータのみ」で学習する。

使い方:
    PYTHONIOENCODING=utf-8 py ml/model/backtest_all_races.py
"""

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.config import DATA_DIR, BASE_DIR

# 基本特徴量（固定: 重要度下位1%を事前除外済み）
# 除外済み: last3_finish, win_rate, mile_win_rate, last2_finish, last1_position_gain
_BASE_FEATURES = [
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

# Model A: 市場連動型（オッズ込み） — 24特徴量
FEATURE_COLS_ALL = _BASE_FEATURES + ["odds", "popularity"]

# Model B: エッジ検出型（オッズ・field_strength除外） — 21特徴量
FEATURE_COLS_NO_ODDS = [
    c for c in _BASE_FEATURES if c != "field_strength"
]

# 後方互換性のため
FEATURE_COLS = FEATURE_COLS_ALL

BLEND_WEIGHT_A = 0.2  # Model A（市場連動型）— 少量のみ
BLEND_WEIGHT_B = 0.8  # Model B（エッジ検出型）— 主力

def _make_params_bin(scale_pos_weight: float = 1.0) -> dict:
    """二値分類パラメータを生成（Optuna最適化済み — Trial#54, 固定特徴量）"""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 39,
        "learning_rate": 0.134,
        "n_estimators": 100,
        "verbose": -1,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        "min_child_samples": 8,
        "reg_alpha": 6.842,
        "reg_lambda": 0.00467,
        "colsample_bytree": 0.512,
        "subsample": 0.652,
        "subsample_freq": 1,
    }


# 後方互換性のため旧名を残す
PARAMS_BIN = _make_params_bin()

# グレード情報
GRADE_MAP = {
    "桜花賞": "G1", "阪神JF": "G1", "オークス": "G1", "秋華賞": "G1",
    "チューリップ賞": "G2", "ローズS": "G2", "フィリーズレビュー": "G2",
    "クイーンC": "G3", "フェアリーS": "G3", "アルテミスS": "G3",
    "ファンタジーS": "G3", "紫苑S": "G3",
}


def extract_year(label: str) -> int:
    """ラベルから年を抽出"""
    m = re.search(r"(\d{4})", label)
    return int(m.group(1)) if m else 0


def get_race_base(label: str) -> str:
    """ラベルからレース基本名を取得"""
    for key in GRADE_MAP:
        if key in label:
            return key
    return "不明"


def build_race_order(df: pd.DataFrame) -> list[dict]:
    """レースを時系列順に並べる（race_id順）"""
    races = []
    for label in df["label"].unique():
        race_df = df[df["label"] == label]
        race_id = race_df["race_id"].iloc[0] if "race_id" in race_df.columns else ""
        year = extract_year(label)
        race_base = get_race_base(label)
        grade = GRADE_MAP.get(race_base, "?")
        races.append({
            "label": label,
            "race_id": str(race_id),
            "year": year,
            "race_base": race_base,
            "grade": grade,
            "n_horses": len(race_df),
        })
    # race_id（=時系列順）でソート
    races.sort(key=lambda x: x["race_id"])
    return races


def _get_available_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    """DataFrameに存在する特徴量のみフィルタ"""
    return [c for c in feature_list if c in df.columns]


def _train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
) -> lgb.LGBMClassifier:
    """早期停止付きLGBMClassifierを学習"""
    model = lgb.LGBMClassifier(**params)
    # 学習データの10%をバリデーションに使用（早期停止用）
    n = len(X_train)
    if n >= 100:
        split = int(n * 0.9)
        model.fit(
            X_train[:split], y_train[:split],
            eval_set=[(X_train[split:], y_train[split:])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        # データが少ない場合は早期停止なし
        model.fit(X_train, y_train)
    return model


def _load_payouts() -> dict:
    """payouts.jsonをロード"""
    payout_path = DATA_DIR / "raw" / "payouts.json"
    if payout_path.exists():
        with open(payout_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _find_payout(payouts_for_race: dict, bet_type: str, combo_set: set[int]) -> int:
    """配当データからcombo_setに一致する配当を検索

    Args:
        payouts_for_race: {"quinella": [...], "wide": [...], "trio": [...]}
        bet_type: "quinella", "wide", "trio"
        combo_set: 馬番のset（例: {1, 5}）

    Returns:
        配当額（円）。見つからなければ0
    """
    entries = payouts_for_race.get(bet_type, [])
    for entry in entries:
        combo_nums = set(int(x) for x in entry["combo"].split("-"))
        if combo_nums == combo_set:
            return entry["payout"]
    return 0


def _fit_calibrators(calib_pairs: list[dict]) -> dict:
    """Isotonic Regressionでキャリブレーターをフィットし、pickle保存 + 曲線データ返却"""
    if not calib_pairs:
        return {}

    pred_win = np.array([p["pred_win"] for p in calib_pairs])
    is_win = np.array([p["is_win"] for p in calib_pairs])
    pred_show = np.array([p["pred_show"] for p in calib_pairs])
    is_show = np.array([p["is_show"] for p in calib_pairs])

    cal_win = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    cal_win.fit(pred_win, is_win)

    cal_show = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    cal_show.fit(pred_show, is_show)

    # pickle保存
    model_dir = Path(__file__).resolve().parent
    with open(model_dir / "calibrator_win.pkl", "wb") as f:
        pickle.dump(cal_win, f)
    with open(model_dir / "calibrator_show.pkl", "wb") as f:
        pickle.dump(cal_show, f)
    print(f"\n✅ キャリブレーター保存: {model_dir}/calibrator_*.pkl")

    # キャリブレーション曲線データ (10ビン)
    def _build_curve(pred: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> list[dict]:
        bins = np.linspace(0, 1, n_bins + 1)
        curve = []
        for i in range(n_bins):
            mask = (pred >= bins[i]) & (pred < bins[i + 1])
            count = int(mask.sum())
            if count == 0:
                continue
            bin_center = round((bins[i] + bins[i + 1]) / 2, 3)
            predicted = round(float(pred[mask].mean()), 4)
            observed = round(float(actual[mask].mean()), 4)
            curve.append({
                "bin_center": bin_center,
                "predicted": predicted,
                "observed": observed,
                "count": count,
            })
        return curve

    return {
        "win": _build_curve(pred_win, is_win),
        "show": _build_curve(pred_show, is_show),
    }


def _bootstrap_confidence(results: list[dict], n_iter: int = 10000) -> dict:
    """Bootstrap法で的中率・ROIの95%信頼区間とp値を算出"""
    n = len(results)
    if n == 0:
        return {}

    rng = np.random.default_rng(42)

    win_hits_arr = np.array([r["win_hit"] for r in results])
    show_hits_arr = np.array([r["show_hits"] for r in results])
    win_return_arr = np.array([r["win_return"] for r in results])
    show_return_arr = np.array([r["show_return"] for r in results])

    bs_win_hit = np.zeros(n_iter)
    bs_win_roi = np.zeros(n_iter)
    bs_show_hit = np.zeros(n_iter)
    bs_show_roi = np.zeros(n_iter)

    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        bs_win_hit[i] = win_hits_arr[idx].mean()
        bs_win_roi[i] = win_return_arr[idx].sum() / (n * 100)
        bs_show_hit[i] = show_hits_arr[idx].mean() / 3  # show_hits is out of 3
        bs_show_roi[i] = show_return_arr[idx].sum() / (n * 300)

    p_value_win_roi = float((bs_win_roi <= 1.0).sum()) / n_iter

    return {
        "win_hit_rate_ci": [
            round(float(np.percentile(bs_win_hit, 2.5)), 3),
            round(float(np.percentile(bs_win_hit, 97.5)), 3),
        ],
        "win_roi_ci": [
            round(float(np.percentile(bs_win_roi, 2.5)), 3),
            round(float(np.percentile(bs_win_roi, 97.5)), 3),
        ],
        "show_hit_rate_ci": [
            round(float(np.percentile(bs_show_hit, 2.5)), 3),
            round(float(np.percentile(bs_show_hit, 97.5)), 3),
        ],
        "show_roi_ci": [
            round(float(np.percentile(bs_show_roi, 2.5)), 3),
            round(float(np.percentile(bs_show_roi, 97.5)), 3),
        ],
        "win_roi_pvalue": round(p_value_win_roi, 4),
    }


def _simulate_bankroll(
    results: list[dict], n_sim: int = 1000, initial: int = 10000,
) -> dict:
    """Monte Carloバンクロールシミュレーション"""
    n = len(results)
    if n == 0:
        return {}

    rng = np.random.default_rng(42)
    kelly_bets = np.array([r["kelly_bet"] for r in results])
    kelly_returns = np.array([r["kelly_return"] for r in results])

    # 各シミュレーションの資金推移を記録
    all_paths = np.zeros((n_sim, n + 1))
    all_paths[:, 0] = initial
    max_drawdowns = np.zeros(n_sim)

    for s in range(n_sim):
        idx = rng.integers(0, n, size=n)
        bankroll = float(initial)
        peak = bankroll
        max_dd = 0.0
        for step in range(n):
            ret = kelly_returns[idx[step]]
            bankroll *= (1 + ret)
            bankroll = max(bankroll, 0)  # 破産防止
            all_paths[s, step + 1] = bankroll
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        max_drawdowns[s] = max_dd

    # パーセンタイルパスを計算
    percentiles = [5, 25, 50, 75, 95]
    paths = {}
    for p in percentiles:
        path = np.percentile(all_paths, p, axis=0)
        paths[f"p{p}"] = [round(float(v)) for v in path]

    final_values = all_paths[:, -1]

    return {
        "initial_bankroll": initial,
        "n_races": n,
        "paths": paths,
        "max_drawdown": {
            "median": round(float(np.median(max_drawdowns)), 3),
            "p95": round(float(np.percentile(max_drawdowns, 95)), 3),
        },
        "final_bankroll": {
            "median": round(float(np.median(final_values))),
            "p5": round(float(np.percentile(final_values, 5))),
            "p95": round(float(np.percentile(final_values, 95))),
        },
    }


def run_walk_forward(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """Walk-Forwardバックテスト（デュアルモデル + 正則化 + 早期停止）"""
    print("=" * 60)
    print("全重賞 Walk-Forward バックテスト（改良版）")
    print("=" * 60)
    print()
    print("方針: 各レースRを予測する際、race_id < R のデータのみで学習")
    print(f"モデル: A(市場連動={BLEND_WEIGHT_A}) + B(エッジ検出={BLEND_WEIGHT_B})")
    print()

    payouts = _load_payouts()
    if payouts:
        print(f"配当データ: {len(payouts)}レース分ロード済み")
    else:
        print("⚠️ payouts.json なし — 組合せ馬券ROIは計算不可")
    print()

    races = build_race_order(df)
    results = []
    min_train_size = 50

    # キャリブレーション用: 全馬の (予測確率, 実績) ペアを蓄積
    calib_pairs: list[dict] = []

    # 利用可能な特徴量を事前確認
    feat_all = _get_available_features(df, FEATURE_COLS_ALL)
    feat_no_odds = _get_available_features(df, FEATURE_COLS_NO_ODDS)

    for i, race in enumerate(races):
        label = race["label"]
        test_df = df[df["label"] == label].copy()

        if test_df.empty:
            continue

        current_race_id = race["race_id"]
        train_df = df[df["race_id"].astype(str) < current_race_id].copy()

        if len(train_df) < min_train_size:
            continue

        # Optuna最適化済み scale_pos_weight（Trial#54）
        params_win = _make_params_bin(scale_pos_weight=11.817)
        params_show = _make_params_bin(scale_pos_weight=6.988)

        X_train_all = train_df[feat_all].values
        X_test_all = test_df[feat_all].values
        X_train_no_odds = train_df[feat_no_odds].values
        X_test_no_odds = test_df[feat_no_odds].values

        y_win = train_df["is_win"].values
        y_show = train_df["is_show"].values

        # --- Model A: 全特徴量（市場連動型） ---
        model_a_win = _train_model(X_train_all, y_win, params_win)
        pred_a_win = model_a_win.predict_proba(X_test_all)[:, 1]

        model_a_show = _train_model(X_train_all, y_show, params_show)
        pred_a_show = model_a_show.predict_proba(X_test_all)[:, 1]

        # --- Model B: オッズ除外（エッジ検出型） ---
        model_b_win = _train_model(X_train_no_odds, y_win, params_win)
        pred_b_win = model_b_win.predict_proba(X_test_no_odds)[:, 1]

        model_b_show = _train_model(X_train_no_odds, y_show, params_show)
        pred_b_show = model_b_show.predict_proba(X_test_no_odds)[:, 1]

        # --- ブレンド ---
        pred_win = BLEND_WEIGHT_A * pred_a_win + BLEND_WEIGHT_B * pred_b_win
        pred_show = BLEND_WEIGHT_A * pred_a_show + BLEND_WEIGHT_B * pred_b_show

        test_df["pred_win"] = pred_win
        test_df["pred_show"] = pred_show
        test_df["pred_b_win"] = pred_b_win  # Model Bの生予測（EV計算用）

        # キャリブレーション用ペアを蓄積
        for _, row in test_df.iterrows():
            calib_pairs.append({
                "pred_win": float(row["pred_win"]),
                "is_win": int(row["着順_num"] == 1),
                "pred_show": float(row["pred_show"]),
                "is_show": int(row["着順_num"] <= 3),
            })

        n_horses = len(test_df)

        # === 評価 ===
        top1 = test_df.nlargest(1, "pred_win").iloc[0]
        actual_winner = test_df[test_df["着順_num"] == 1]
        winner_name = actual_winner["馬名"].values[0] if len(actual_winner) > 0 else "?"
        win_hit = int(top1["馬名"] == winner_name)

        top3 = test_df.nlargest(3, "pred_show")
        actual_top3 = set(test_df[test_df["着順_num"] <= 3]["馬名"].values)
        pred_top3 = set(top3["馬名"].values)
        show_hits = len(actual_top3 & pred_top3)

        fav = test_df.loc[test_df["popularity"].idxmin()] if test_df["popularity"].notna().any() else None
        fav_won = int(fav["着順_num"] == 1) if fav is not None else 0
        fav_show = int(fav["着順_num"] <= 3) if fav is not None else 0

        win_return = float(top1["odds"]) * 100 if top1["着順_num"] == 1 else 0.0

        show_return_total = 0.0
        for _, row in top3.iterrows():
            if row["着順_num"] <= 3:
                show_return_total += row["odds"] * 0.3 * 100

        # === 組合せ馬券的中判定 ===
        actual_top2_nums = set(test_df[test_df["着順_num"] <= 2]["horse_number"].values)
        actual_top3_nums = set(test_df[test_df["着順_num"] <= 3]["horse_number"].values)
        pred_top3_nums = set(test_df.nlargest(3, "pred_win")["horse_number"].values)
        pred_top5_nums = set(test_df.nlargest(5, "pred_win")["horse_number"].values)
        pred_top2_nums = set(test_df.nlargest(2, "pred_win")["horse_number"].values)

        # 馬連BOX(3): AI上位3頭のうち2頭が1-2着
        quinella_box3_hit = int(len(pred_top3_nums & actual_top2_nums) >= 2)
        # ワイド(上位2頭): AI上位2頭が両方3着以内
        wide_top2_hit = int(len(pred_top2_nums & actual_top3_nums) >= 2)
        # 三連複BOX(3): AI上位3頭が全員3着以内
        trio_box3_hit = int(pred_top3_nums <= actual_top3_nums)
        # 三連複BOX(5): AI上位5頭のうち3頭が1-2-3着
        trio_box5_hit = int(len(pred_top5_nums & actual_top3_nums) >= 3)

        # === 組合せ馬券ROI計算（実配当ベース） ===
        race_payouts = payouts.get(current_race_id, {})
        quinella_box3_roi = 0.0
        wide_top2_roi = 0.0
        trio_box3_roi = 0.0
        trio_box5_roi = 0.0

        if race_payouts:
            # 馬連BOX(3): 3C2=3通り×100円=300円投資
            if quinella_box3_hit:
                hit_combo = actual_top2_nums  # 実際の1-2着馬番
                payout = _find_payout(race_payouts, "quinella", hit_combo)
                quinella_box3_roi = payout / 300.0  # 配当/投資額

            # ワイド(◎-○): 1通り×100円=100円投資
            if wide_top2_hit:
                payout = _find_payout(race_payouts, "wide", pred_top2_nums)
                wide_top2_roi = payout / 100.0

            # 三連複BOX(3): 1通り×100円=100円投資
            if trio_box3_hit:
                payout = _find_payout(race_payouts, "trio", actual_top3_nums)
                trio_box3_roi = payout / 100.0

            # 三連複BOX(5): 5C3=10通り×100円=1000円投資
            if trio_box5_hit:
                payout = _find_payout(race_payouts, "trio", actual_top3_nums)
                trio_box5_roi = payout / 1000.0

        # Kelly return (1/4 Kelly): AI本命馬の単勝Kelly
        ai_odds = float(top1["odds"])
        ai_prob = float(top1["pred_win"])
        kelly_frac = 0.25
        if ai_odds > 1.0 and ai_prob > 0:
            b = ai_odds - 1
            f_full = (ai_prob * b - (1 - ai_prob)) / b
            kelly_bet = max(0.0, f_full * kelly_frac)
        else:
            kelly_bet = 0.0
        # 的中時: +kelly * (odds-1), 不的中時: -kelly
        if win_hit and kelly_bet > 0:
            kelly_return = kelly_bet * (ai_odds - 1)
        else:
            kelly_return = -kelly_bet

        results.append({
            "label": label,
            "race_base": race["race_base"],
            "grade": race["grade"],
            "year": race["year"],
            "n_horses": n_horses,
            "train_size": len(train_df),
            "ai_pick": str(top1["馬名"]),
            "actual_winner": winner_name,
            "win_hit": win_hit,
            "show_hits": show_hits,
            "fav_won": fav_won,
            "fav_show": fav_show,
            "win_return": win_return,
            "show_return": show_return_total,
            "ai_odds": round(float(top1["odds"]), 1),
            "kelly_bet": round(kelly_bet, 6),
            "kelly_return": round(kelly_return, 6),
            "quinella_box3_hit": quinella_box3_hit,
            "wide_top2_hit": wide_top2_hit,
            "trio_box3_hit": trio_box3_hit,
            "trio_box5_hit": trio_box5_hit,
            "quinella_box3_roi": quinella_box3_roi,
            "wide_top2_roi": wide_top2_roi,
            "trio_box3_roi": trio_box3_roi,
            "trio_box5_roi": trio_box5_roi,
        })

    # --- キャリブレーター生成 & pickle保存 ---
    calib_data = _fit_calibrators(calib_pairs)

    return results, calib_data


def print_summary(
    results: list[dict],
    calib_data: dict | None = None,
    df: pd.DataFrame | None = None,
) -> None:
    """サマリー出力"""
    n = len(results)
    if n == 0:
        print("結果なし")
        return

    print()
    print("=" * 65)
    print(f"全体サマリー（{n}レース）")
    print("=" * 65)

    # --- 全体 ---
    win_hits = sum(r["win_hit"] for r in results)
    show_hits_total = sum(r["show_hits"] for r in results)
    show_possible = n * 3
    fav_wins = sum(r["fav_won"] for r in results)
    fav_shows = sum(r["fav_show"] for r in results)

    total_win_invest = n * 100
    total_win_return = sum(r["win_return"] for r in results)
    win_roi = total_win_return / total_win_invest

    total_show_invest = n * 3 * 100
    total_show_return = sum(r["show_return"] for r in results)
    show_roi = total_show_return / total_show_invest

    avg_horses = np.mean([r["n_horses"] for r in results])

    # --- 組合せ馬券 ---
    quinella_hits = sum(r.get("quinella_box3_hit", 0) for r in results)
    wide_hits = sum(r.get("wide_top2_hit", 0) for r in results)
    trio3_hits = sum(r.get("trio_box3_hit", 0) for r in results)
    trio5_hits = sum(r.get("trio_box5_hit", 0) for r in results)

    # 組合せ馬券ROI（投資額ベース）
    quinella_invest = n * 300  # 3通り×100円
    quinella_return = sum(r.get("quinella_box3_roi", 0) * 300 for r in results)
    quinella_roi = quinella_return / quinella_invest if quinella_invest > 0 else 0

    wide_invest = n * 100  # 1通り×100円
    wide_return = sum(r.get("wide_top2_roi", 0) * 100 for r in results)
    wide_roi = wide_return / wide_invest if wide_invest > 0 else 0

    trio3_invest = n * 100  # 1通り×100円
    trio3_return = sum(r.get("trio_box3_roi", 0) * 100 for r in results)
    trio3_roi = trio3_return / trio3_invest if trio3_invest > 0 else 0

    trio5_invest = n * 1000  # 10通り×100円
    trio5_return = sum(r.get("trio_box5_roi", 0) * 1000 for r in results)
    trio5_roi = trio5_return / trio5_invest if trio5_invest > 0 else 0

    print(f"\n平均出走頭数: {avg_horses:.1f}頭")
    print(f"\n{'':>20} {'AI':>10} {'1番人気':>10} {'ランダム':>10}")
    print("-" * 55)
    print(f"{'1着的中率':>20} {win_hits/n:>9.1%} {fav_wins/n:>9.1%} {1/avg_horses:>9.1%}")
    print(f"{'複勝的中率':>20} {show_hits_total/show_possible:>9.1%} {fav_shows/n:>9.1%} {3/avg_horses:>9.1%}")
    print(f"{'単勝回収率':>20} {win_roi:>9.0%} {'---':>10} {'---':>10}")
    print(f"{'複勝回収率':>20} {show_roi:>9.0%} {'---':>10} {'---':>10}")

    print(f"\n{'組合せ馬券':>20} {'的中率':>10} {'回収率':>10}")
    print("-" * 45)
    print(f"  馬連BOX(3):   {quinella_hits}/{n} ({quinella_hits/n:>4.0%})  {quinella_roi:>6.0%}")
    print(f"  ワイド(◎-○):  {wide_hits}/{n} ({wide_hits/n:>4.0%})  {wide_roi:>6.0%}")
    print(f"  三連複BOX(3): {trio3_hits}/{n} ({trio3_hits/n:>4.0%})  {trio3_roi:>6.0%}")
    print(f"  三連複BOX(5): {trio5_hits}/{n} ({trio5_hits/n:>4.0%})  {trio5_roi:>6.0%}")

    # --- グレード別 ---
    print(f"\n{'='*65}")
    print("グレード別")
    print(f"{'='*65}")

    for grade in ["G1", "G2", "G3"]:
        gr = [r for r in results if r["grade"] == grade]
        if not gr:
            continue
        gn = len(gr)
        g_win = sum(r["win_hit"] for r in gr)
        g_show = sum(r["show_hits"] for r in gr)
        g_fav_w = sum(r["fav_won"] for r in gr)
        g_win_inv = gn * 100
        g_win_ret = sum(r["win_return"] for r in gr)
        g_show_inv = gn * 3 * 100
        g_show_ret = sum(r["show_return"] for r in gr)

        print(f"\n  [{grade}] {gn}レース")
        print(f"    1着的中: {g_win}/{gn} ({g_win/gn:.0%})  1番人気: {g_fav_w}/{gn} ({g_fav_w/gn:.0%})")
        print(f"    複勝的中: {g_show}/{gn*3} ({g_show/(gn*3):.0%})")
        print(f"    単勝回収率: {g_win_ret/g_win_inv:.0%}  複勝回収率: {g_show_ret/g_show_inv:.0%}")

    # --- レース別 ---
    print(f"\n{'='*65}")
    print("レース別")
    print(f"{'='*65}")

    race_bases = sorted(set(r["race_base"] for r in results))
    for rb in race_bases:
        rr = [r for r in results if r["race_base"] == rb]
        rn = len(rr)
        r_win = sum(r["win_hit"] for r in rr)
        r_show = sum(r["show_hits"] for r in rr)
        r_win_inv = rn * 100
        r_win_ret = sum(r["win_return"] for r in rr)
        r_show_inv = rn * 3 * 100
        r_show_ret = sum(r["show_return"] for r in rr)
        grade = rr[0]["grade"]

        print(f"  {rb:>12}({grade}) {rn}R  "
              f"1着{r_win}/{rn}({r_win/rn:>4.0%}) "
              f"複勝{r_show}/{rn*3}({r_show/(rn*3):>4.0%}) "
              f"単回{r_win_ret/r_win_inv:>4.0%} "
              f"複回{r_show_ret/r_show_inv:>4.0%}")

    # --- 年別 ---
    print(f"\n{'='*65}")
    print("年別推移")
    print(f"{'='*65}")

    for year in sorted(set(r["year"] for r in results)):
        yr = [r for r in results if r["year"] == year]
        yn = len(yr)
        y_win = sum(r["win_hit"] for r in yr)
        y_show = sum(r["show_hits"] for r in yr)
        y_win_inv = yn * 100
        y_win_ret = sum(r["win_return"] for r in yr)
        y_show_inv = yn * 3 * 100
        y_show_ret = sum(r["show_return"] for r in yr)

        print(f"  {year}: {yn}R  "
              f"1着{y_win}/{yn}({y_win/yn:>4.0%}) "
              f"複勝{y_show}/{yn*3}({y_show/(yn*3):>4.0%}) "
              f"単回{y_win_ret/y_win_inv:>4.0%} "
              f"複回{y_show_ret/y_show_inv:>4.0%}")

    # --- 個別レース一覧 ---
    print(f"\n{'='*65}")
    print("個別レース一覧")
    print(f"{'='*65}")
    print(f"{'レース':>18} {'AI本命':>10} {'1着':>3} {'複勝':>4} {'オッズ':>5} {'回収':>6}")
    print("-" * 65)
    for r in results:
        print(
            f"  {r['label']:>16} "
            f"{r['ai_pick']:>10} "
            f"{'◯' if r['win_hit'] else '✗':>3} "
            f"{r['show_hits']}/3  "
            f"{r['ai_odds']:>4.1f} "
            f"{'◯' + str(int(r['win_return'])) + '円' if r['win_return'] > 0 else '  ---':>6}"
        )

    # --- 結論 ---
    print(f"\n{'='*65}")
    print("結論")
    print(f"{'='*65}")
    print(f"  評価レース数: {n}")
    print(f"  単勝回収率: {win_roi:.0%} ({'プラス' if win_roi >= 1 else 'マイナス'})")
    print(f"  複勝回収率: {show_roi:.0%} ({'プラス' if show_roi >= 1 else 'マイナス'})")
    if show_roi >= 1:
        profit = total_show_return - total_show_invest
        print(f"  → 複勝戦略で {profit:,.0f}円のプラス（{n}レース通算）")
    print(f"\n  ※ JRA控除率: 単勝20%, 複勝20%")
    print(f"  ※ 回収率80%以上で「市場平均超え」、100%超で「プラス収支」")

    # --- Bootstrap信頼区間 ---
    confidence = _bootstrap_confidence(results)
    if confidence:
        print(f"\n{'='*65}")
        print("Bootstrap 95% 信頼区間 (10,000回)")
        print(f"{'='*65}")
        ci = confidence
        print(f"  1着的中率: [{ci['win_hit_rate_ci'][0]:.1%} – {ci['win_hit_rate_ci'][1]:.1%}]")
        print(f"  複勝的中率: [{ci['show_hit_rate_ci'][0]:.1%} – {ci['show_hit_rate_ci'][1]:.1%}]")
        print(f"  単勝回収率: [{ci['win_roi_ci'][0]:.0%} – {ci['win_roi_ci'][1]:.0%}]")
        print(f"  複勝回収率: [{ci['show_roi_ci'][0]:.0%} – {ci['show_roi_ci'][1]:.0%}]")
        print(f"  単勝ROI > 100% p値: {ci['win_roi_pvalue']:.4f}")

    # --- バンクロールシミュレーション ---
    simulation = _simulate_bankroll(results)
    if simulation:
        print(f"\n{'='*65}")
        print("バンクロールシミュレーション (1,000パス)")
        print(f"{'='*65}")
        fb = simulation["final_bankroll"]
        mdd = simulation["max_drawdown"]
        print(f"  初期資金: ¥{simulation['initial_bankroll']:,}")
        print(f"  最終資金 中央値: ¥{fb['median']:,}")
        print(f"  最終資金 5%tile: ¥{fb['p5']:,}  95%tile: ¥{fb['p95']:,}")
        print(f"  最大DD 中央値: {mdd['median']:.1%}  95%tile: {mdd['p95']:.1%}")

    # --- JSON保存 ---
    output = {
        "summary": {
            "n_races": n,
            "win_hit_rate": round(win_hits / n, 3),
            "show_hit_rate": round(show_hits_total / show_possible, 3),
            "win_roi": round(win_roi, 3),
            "show_roi": round(show_roi, 3),
            "fav_win_rate": round(fav_wins / n, 3),
            "fav_show_rate": round(fav_shows / n, 3),
            "confidence": confidence if confidence else None,
        },
        "combo_hit_rates": {
            "quinella_box3": round(quinella_hits / n, 3) if n > 0 else 0,
            "wide_top2": round(wide_hits / n, 3) if n > 0 else 0,
            "trio_box3": round(trio3_hits / n, 3) if n > 0 else 0,
            "trio_box5": round(trio5_hits / n, 3) if n > 0 else 0,
            "quinella_box3_roi": round(quinella_roi, 3),
            "wide_top2_roi": round(wide_roi, 3),
            "trio_box3_roi": round(trio3_roi, 3),
            "trio_box5_roi": round(trio5_roi, 3),
        },
        "by_grade": {},
        "by_race": {},
        "by_year": {},
        "races": [],
    }

    for grade in ["G1", "G2", "G3"]:
        gr = [r for r in results if r["grade"] == grade]
        if gr:
            gn = len(gr)
            output["by_grade"][grade] = {
                "n": gn,
                "win_rate": round(sum(r["win_hit"] for r in gr) / gn, 3),
                "show_rate": round(sum(r["show_hits"] for r in gr) / (gn * 3), 3),
                "win_roi": round(sum(r["win_return"] for r in gr) / (gn * 100), 3),
                "show_roi": round(sum(r["show_return"] for r in gr) / (gn * 300), 3),
            }

    for rb in race_bases:
        rr = [r for r in results if r["race_base"] == rb]
        rn = len(rr)
        output["by_race"][rb] = {
            "n": rn,
            "grade": rr[0]["grade"],
            "win_rate": round(sum(r["win_hit"] for r in rr) / rn, 3),
            "show_rate": round(sum(r["show_hits"] for r in rr) / (rn * 3), 3),
            "win_roi": round(sum(r["win_return"] for r in rr) / (rn * 100), 3),
            "show_roi": round(sum(r["show_return"] for r in rr) / (rn * 300), 3),
        }

    for year in sorted(set(r["year"] for r in results)):
        yr = [r for r in results if r["year"] == year]
        yn = len(yr)
        output["by_year"][str(year)] = {
            "n": yn,
            "win_rate": round(sum(r["win_hit"] for r in yr) / yn, 3),
            "show_rate": round(sum(r["show_hits"] for r in yr) / (yn * 3), 3),
            "win_roi": round(sum(r["win_return"] for r in yr) / (yn * 100), 3),
            "show_roi": round(sum(r["show_return"] for r in yr) / (yn * 300), 3),
        }

    for r in results:
        output["races"].append({
            "label": r["label"],
            "grade": r["grade"],
            "year": r["year"],
            "ai_pick": r["ai_pick"],
            "actual_winner": r["actual_winner"],
            "win_hit": r["win_hit"],
            "show_hits": r["show_hits"],
            "win_return": r["win_return"],
            "show_return": round(r["show_return"], 1),
        })

    # 統計データ（枠順別・人気別・血統別）
    if df is not None:
        try:
            # 枠順別勝率
            frame_stats = []
            for f in sorted(df["frame_number"].dropna().unique()):
                sub = df[df["frame_number"] == f]
                wins = int(sub["is_win"].sum())
                total = len(sub)
                frame_stats.append({
                    "frame": f"{int(f)}枠",
                    "rate": round(wins / total * 100, 1) if total > 0 else 0,
                    "n": total,
                })
            output["frame_win_rate"] = frame_stats

            # 人気別3着内率
            pop_stats = []
            for p in range(1, 11):
                sub = df[df["popularity"] == p]
                shows = int(sub["is_show"].sum())
                total = len(sub)
                if total > 0:
                    pop_stats.append({
                        "pop": f"{p}人気",
                        "rate": round(shows / total * 100, 1),
                        "n": total,
                    })
            output["popularity_show_rate"] = pop_stats

            # 血統カテゴリ別勝率
            pedigree_path = DATA_DIR / "raw" / "horse_pedigree.csv"
            if pedigree_path.exists():
                ped_df = pd.read_csv(pedigree_path)
                merged = df.merge(
                    ped_df[["horse_name", "sire"]],
                    left_on="馬名", right_on="horse_name", how="left",
                )
                # カテゴリ代表名を取得
                sire_stats = []
                for code in sorted(merged["sire_category_code"].dropna().unique()):
                    sub = merged[merged["sire_category_code"] == code]
                    wins = int(sub["is_win"].sum())
                    total = len(sub)
                    # 最頻出の父を代表名に
                    top_sire = sub["sire"].value_counts().index[0] if sub["sire"].notna().any() else f"Cat{int(code)}"
                    sire_stats.append({
                        "name": f"{top_sire}系",
                        "rate": round(wins / total * 100, 1) if total > 0 else 0,
                        "n": total,
                    })
                # 勝率降順
                sire_stats.sort(key=lambda x: -x["rate"])
                output["bloodline_win_rate"] = sire_stats
        except Exception as e:
            print(f"  ⚠️ 統計データ生成失敗: {e}")

    # 特徴量重要度（実モデルから取得）
    if df is not None:
        try:
            importance = get_feature_importance(df)
            total_imp = sum(importance.values())
            if total_imp > 0:
                # 日本語ラベルマッピング
                label_map = {
                    "speed_index": "スピード指数",
                    "weight": "馬体重",
                    "field_strength": "場の強さ",
                    "odds": "オッズ",
                    "horse_number": "馬番",
                    "frame_number": "枠番",
                    "weight_diff": "馬体重増減",
                    "jockey_win_rate": "騎手勝率",
                    "popularity": "人気",
                    "last1_start_pos": "前走スタート位置",
                    "last1_speed": "前走スピード",
                    "trainer_win_rate": "調教師勝率",
                    "best_last3f": "最速上がり3F",
                    "avg_last3f": "平均上がり3F",
                    "grade_encoded": "グレード",
                    "last1_last3f": "前走上がり3F",
                    "last2_last3f": "2走前上がり3F",
                    "total_runs": "通算出走数",
                    "show_rate": "複勝率",
                    "last1_finish": "前走着順",
                    "hanshin_runs": "阪神実績",
                    "jockey_g1_wins": "騎手G1勝数",
                    "distance_m": "距離",
                    "last1_margin": "前走着差",
                }
                sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:10]
                output["feature_importance"] = [
                    {
                        "name": label_map.get(name, name),
                        "key": name,
                        "value": round(val / total_imp, 4),
                    }
                    for name, val in sorted_imp
                ]
        except Exception as e:
            print(f"  ⚠️ 特徴量重要度取得失敗: {e}")

    # キャリブレーション・シミュレーションデータ追加
    if calib_data:
        output["calibration"] = calib_data
    if simulation:
        output["simulation"] = simulation

    json_path = BASE_DIR / "frontend" / "src" / "data" / "backtest_all.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ backtest_all.json 保存: {json_path}")


def get_feature_importance(df: pd.DataFrame) -> dict[str, float]:
    """全学習データでモデルを1回学習し、特徴量重要度を取得"""
    feat_all = _get_available_features(df, FEATURE_COLS_ALL)
    X = df[feat_all].values
    y = df["is_win"].values

    params = _make_params_bin(
        scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1)
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    importance = dict(zip(feat_all, model.feature_importances_))
    return importance


def select_features(
    importance: dict[str, float],
    threshold_pct: float = 1.0,
) -> tuple[list[str], list[str]]:
    """重要度下位 threshold_pct% の特徴量を除外

    Returns:
        (kept_features, dropped_features)
    """
    total = sum(importance.values())
    if total == 0:
        return list(importance.keys()), []

    sorted_feats = sorted(importance.items(), key=lambda x: x[1])
    cumsum = 0.0
    dropped = []
    for name, imp in sorted_feats:
        pct = imp / total * 100
        cumsum += pct
        if cumsum <= threshold_pct:
            dropped.append(name)
        else:
            break

    kept = [f for f in importance if f not in dropped]
    return kept, dropped


def main() -> None:
    csv_path = DATA_DIR / "features.csv"
    if not csv_path.exists():
        print("❌ features.csv が見つかりません")
        return

    df = pd.read_csv(csv_path)
    print(f"入力: {len(df)}行 × {len(df.columns)}カラム\n")

    # 特徴量は _BASE_FEATURES に固定（重要度下位1%は事前除外済み）
    print(f"特徴量（固定）: Model A {len(FEATURE_COLS_ALL)}個 / Model B {len(FEATURE_COLS_NO_ODDS)}個\n")

    # 本番バックテスト
    results, calib_data = run_walk_forward(df)
    print_summary(results, calib_data, df)
    print("\n完了!")


if __name__ == "__main__":
    main()
