"""
SAKURA ORACLE — LightGBM予測モデル＆predictions.json生成

使い方:
    PYTHONIOENCODING=utf-8 py ml/model/predictor.py

入力: data/features.csv
出力: frontend/src/data/predictions.json（本番データで上書き）
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.config import DATA_DIR, BASE_DIR

# === 設定 ===
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
    # ペース（新規）
    "pace_deviation", "last1_pace_deviation", "n_front_runners_est",
    # 出走間隔
    "rest_weeks",
]

# Model A: 市場連動型（オッズ込み）
FEATURE_COLS_ALL = _BASE_FEATURES + ["odds", "popularity"]

# Model B: エッジ検出型（オッズ・field_strength除外）
FEATURE_COLS_NO_ODDS = [
    c for c in _BASE_FEATURES if c != "field_strength"
]

# 後方互換性のため
FEATURE_COLS = FEATURE_COLS_ALL

BLEND_WEIGHT_A = 0.2  # Model A（市場連動型）— 少量のみ
BLEND_WEIGHT_B = 0.8  # Model B（エッジ検出型）— 主力

SAKURA_LABELS = [
    "桜花賞2025", "桜花賞2024", "桜花賞2023", "桜花賞2022", "桜花賞2021",
]

# レーダーチャート用の特徴量マッピング
RADAR_FEATURES = {
    "speed": "speed_index",
    "stamina": "total_runs",
    "instant": "best_last3f",
    "pedigree": "sire_category_code",
    "jockey": "jockey_win_rate",
    "course_fit": "hanshin_runs",
}


def _get_available_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    """DataFrameに存在する特徴量のみフィルタ"""
    return [c for c in feature_list if c in df.columns]


def _train_lgbm_cls(
    X_train: np.ndarray, y_train: np.ndarray,
    params: dict,
) -> lgb.LGBMClassifier:
    """早期停止付きLGBMClassifierを学習"""
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


def _make_params_bin(scale_pos_weight: float = 1.0) -> dict:
    """二値分類パラメータ（桜花賞専用 = G1パラメータ — Trial#87）"""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 40,
        "learning_rate": 0.149,
        "n_estimators": 100,
        "verbose": -1,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        "min_child_samples": 23,
        "reg_alpha": 1.059,
        "reg_lambda": 3.483,
        "colsample_bytree": 0.884,
        "subsample": 0.601,
        "subsample_freq": 6,
    }


def train_models(df: pd.DataFrame) -> dict:
    """デュアルLightGBMモデルを学習（桜花賞をテストセット）"""
    print("\n--- モデル学習（デュアルモデル + 正則化 + 早期停止） ---")

    sakura_mask = df["race_base"] == "桜花賞"
    train_df = df[~sakura_mask].copy()
    test_df = df[sakura_mask].copy()

    feat_all = _get_available_features(df, FEATURE_COLS_ALL)
    feat_no_odds = _get_available_features(df, FEATURE_COLS_NO_ODDS)

    X_train_all = train_df[feat_all].values
    X_test_all = test_df[feat_all].values
    X_train_no_odds = train_df[feat_no_odds].values
    X_test_no_odds = test_df[feat_no_odds].values

    # Nested CV最適化済み scale_pos_weight（Trial#87）
    params_win = _make_params_bin(scale_pos_weight=16.851)
    params_show = _make_params_bin(scale_pos_weight=5.020)

    models = {}

    # === 着順予測（回帰） ===
    params_reg = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "verbose": -1,
        "random_state": 42,
        "min_child_samples": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "subsample_freq": 5,
    }
    model_finish = lgb.LGBMRegressor(**params_reg)
    n = len(X_train_all)
    if n >= 100:
        split = int(n * 0.9)
        model_finish.fit(
            X_train_all[:split], train_df["着順_num"].values[:split],
            eval_set=[(X_train_all[split:], train_df["着順_num"].values[split:])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        model_finish.fit(X_train_all, train_df["着順_num"].values)
    pred_finish = model_finish.predict(X_test_all)
    rmse = np.sqrt(mean_squared_error(test_df["着順_num"].values, pred_finish))
    print(f"  着順予測 RMSE: {rmse:.3f}")
    models["finish"] = model_finish

    y_win = train_df["is_win"].values
    y_show = train_df["is_show"].values

    # === Model A: 全特徴量（市場連動型） ===
    model_a_win = _train_lgbm_cls(X_train_all, y_win, params_win)
    pred_a_win = model_a_win.predict_proba(X_test_all)[:, 1]
    print(f"  Model A 勝率 LogLoss: {log_loss(test_df['is_win'].values, pred_a_win):.4f}")
    models["win_a"] = model_a_win

    model_a_show = _train_lgbm_cls(X_train_all, y_show, params_show)
    pred_a_show = model_a_show.predict_proba(X_test_all)[:, 1]
    print(f"  Model A 複勝 LogLoss: {log_loss(test_df['is_show'].values, pred_a_show):.4f}")
    models["show_a"] = model_a_show

    # === Model B: オッズ除外（エッジ検出型） ===
    model_b_win = _train_lgbm_cls(X_train_no_odds, y_win, params_win)
    pred_b_win = model_b_win.predict_proba(X_test_no_odds)[:, 1]
    print(f"  Model B 勝率 LogLoss: {log_loss(test_df['is_win'].values, pred_b_win):.4f}")
    models["win_b"] = model_b_win

    model_b_show = _train_lgbm_cls(X_train_no_odds, y_show, params_show)
    pred_b_show = model_b_show.predict_proba(X_test_no_odds)[:, 1]
    print(f"  Model B 複勝 LogLoss: {log_loss(test_df['is_show'].values, pred_b_show):.4f}")
    models["show_b"] = model_b_show

    # ブレンド予測
    pred_win = BLEND_WEIGHT_A * pred_a_win + BLEND_WEIGHT_B * pred_b_win
    pred_show = BLEND_WEIGHT_A * pred_a_show + BLEND_WEIGHT_B * pred_b_show
    print(f"\n  ブレンド勝率 LogLoss: {log_loss(test_df['is_win'].values, pred_win):.4f}")
    print(f"  ブレンド複勝 LogLoss: {log_loss(test_df['is_show'].values, pred_show):.4f}")

    # 特徴量重要度（Model A - Top10）
    importance = model_a_win.feature_importances_
    feat_imp = sorted(zip(feat_all, importance), key=lambda x: -x[1])
    print("\n  特徴量重要度 Top10 (Model A):")
    for name, imp in feat_imp[:10]:
        print(f"    {name}: {imp}")

    # Model B の重要度も表示
    importance_b = model_b_win.feature_importances_
    feat_imp_b = sorted(zip(feat_no_odds, importance_b), key=lambda x: -x[1])
    print("\n  特徴量重要度 Top10 (Model B - オッズなし):")
    for name, imp in feat_imp_b[:10]:
        print(f"    {name}: {imp}")

    # 後方互換性のためのエイリアス
    models["win"] = model_a_win
    models["show"] = model_a_show
    models["feat_all"] = feat_all
    models["feat_no_odds"] = feat_no_odds

    return models


def backtest(df: pd.DataFrame, models: dict) -> list[dict]:
    """過去5年の桜花賞バックテスト（デュアルモデル対応）"""
    print("\n--- バックテスト ---")
    results = []

    feat_all = models.get("feat_all", _get_available_features(df, FEATURE_COLS_ALL))
    feat_no_odds = models.get("feat_no_odds", _get_available_features(df, FEATURE_COLS_NO_ODDS))

    for label in SAKURA_LABELS:
        race_df = df[df["label"] == label].copy()
        if race_df.empty:
            continue

        X_all = race_df[feat_all].values
        X_no_odds = race_df[feat_no_odds].values
        pred_a_show = models["show_a"].predict_proba(X_all)[:, 1]
        pred_b_show = models["show_b"].predict_proba(X_no_odds)[:, 1]
        race_df = race_df.copy()
        race_df["pred_show"] = BLEND_WEIGHT_A * pred_a_show + BLEND_WEIGHT_B * pred_b_show

        # 予測上位3頭
        top3_pred = race_df.nlargest(3, "pred_show")
        # 実際の3着以内
        actual_show = set(race_df[race_df["着順_num"] <= 3]["馬名"].values)
        predicted_show = set(top3_pred["馬名"].values)
        hit = len(actual_show & predicted_show)

        year = label.replace("桜花賞", "")
        print(f"  {year}: {hit}/3頭的中 (予測: {list(predicted_show)[:3]})")
        results.append({"year": year, "hit": hit, "total": 3})

    return results


def normalize_0_100(series: pd.Series) -> pd.Series:
    """0-100に正規化（NaN対応）"""
    s = series.fillna(series.median() if series.notna().any() else 50)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(50, index=s.index)
    return ((s - mn) / (mx - mn) * 100).round(0).clip(0, 100).astype(int)


def calc_kelly(prob: float, odds: float, fraction: float = 0.25) -> float:
    """fractional Kelly基準で最適賭け比率を計算

    Args:
        prob: モデル予測勝率
        odds: 単勝オッズ（日本式: 賭け金込みの払い戻し倍率）
        fraction: Kelly fraction（0.25 = 1/4 Kelly、モデル過信防止）

    Returns:
        0以上の賭け比率（資金に対する割合）。エッジなしなら0。
    """
    if odds <= 1.0 or prob <= 0 or prob >= 1:
        return 0.0
    b = odds - 1  # net odds（利益部分）
    # Kelly: f = (p*b - q) / b  where q = 1 - p
    f = (prob * b - (1 - prob)) / b
    return max(0.0, f * fraction)


def get_mark(row: pd.Series, df: pd.DataFrame) -> str:
    """Kelly基準に基づく印付与（1/4 Kelly値域に対応）"""
    kelly = row.get("kelly_win", 0)
    kelly_rank = df["kelly_win"].rank(ascending=False)
    idx = row.name

    if kelly_rank[idx] == 1 and kelly > 0.01:
        return "◎"  # Kelly最大 & 十分なエッジ
    elif kelly_rank[idx] <= 3 and kelly > 0.005:
        return "○"  # 上位3位 & エッジあり
    elif kelly_rank[idx] <= 8 and kelly > 0.002:
        return "▲"  # 上位8位以内 & Kelly正だが控えめ
    # △: AI複勝50% + 市場複勝50%のブレンドで判定
    # LightGBMの分解能不足で同一show_probが多発 → 市場情報で差別化
    blended = row.get("_blended_show", row["pred_show"])
    if blended >= 0.2:
        return "△"  # 複勝候補（3着以内の可能性が高い）
    else:
        return "×"  # エッジなし


def generate_comment(row: pd.Series) -> str:
    """テンプレートベースでAIコメント生成"""
    parts = []

    # 前走情報
    if not pd.isna(row.get("last1_finish")):
        finish = int(row["last1_finish"])
        parts.append(f"前走{finish}着")

    # 上がり3F
    if not pd.isna(row.get("last1_last3f")):
        l3f = row["last1_last3f"]
        if l3f < 34.0:
            parts.append(f"上がり{l3f}秒は優秀")
        elif l3f < 35.0:
            parts.append(f"上がり{l3f}秒は平均的")

    # 阪神実績
    if row.get("hanshin_runs", 0) > 0:
        parts.append(f"阪神{int(row['hanshin_runs'])}走経験あり")

    # 勝率
    if row.get("win_rate", 0) > 0.3:
        parts.append("高い勝率が魅力")
    elif row.get("show_rate", 0) > 0.5:
        parts.append("安定した成績")

    return "。".join(parts) + "。" if parts else "データ不足のため分析困難。"


def generate_risk(row: pd.Series, venue: str = "阪神") -> str:
    """リスク要因を生成

    Args:
        row: 馬データの行
        venue: レース開催会場名（デフォルト: 阪神）
    """
    risks = []
    if row.get("total_runs", 0) < 3:
        risks.append("キャリアが浅い")
    if row.get("hanshin_runs", 0) == 0:
        risks.append(f"{venue}未経験")
    if row.get("last1_finish", 99) > 5:
        risks.append("前走着順が振るわない")
    if row.get("weight_diff", 0) > 10:
        risks.append("馬体重増が大きい")
    return "。".join(risks) if risks else "特に目立つリスクなし"


def generate_predictions_json(
    df: pd.DataFrame, models: dict, backtest_results: list[dict]
) -> None:
    """2025年桜花賞の予測JSONを生成（デュアルモデルブレンド）"""
    print("\n--- predictions.json 生成（デュアルモデル） ---")

    race_df = df[df["label"] == "桜花賞2025"].copy()
    if race_df.empty:
        print("⚠️ 桜花賞2025のデータがありません")
        return

    # 血統データ読み込み（sire表示用）
    pedigree_path = DATA_DIR / "raw" / "horse_pedigree.csv"
    sire_lookup: dict[str, str] = {}
    if pedigree_path.exists():
        ped_df = pd.read_csv(pedigree_path)
        sire_lookup = dict(zip(ped_df["horse_name"], ped_df["sire"]))

    feat_all = models.get("feat_all", _get_available_features(df, FEATURE_COLS_ALL))
    feat_no_odds = models.get("feat_no_odds", _get_available_features(df, FEATURE_COLS_NO_ODDS))

    X_all = race_df[feat_all].values
    X_no_odds = race_df[feat_no_odds].values

    # デュアルモデル予測
    pred_a_win = models["win_a"].predict_proba(X_all)[:, 1]
    pred_b_win = models["win_b"].predict_proba(X_no_odds)[:, 1]
    race_df["pred_win"] = BLEND_WEIGHT_A * pred_a_win + BLEND_WEIGHT_B * pred_b_win

    pred_a_show = models["show_a"].predict_proba(X_all)[:, 1]
    pred_b_show = models["show_b"].predict_proba(X_no_odds)[:, 1]
    race_df["pred_show"] = BLEND_WEIGHT_A * pred_a_show + BLEND_WEIGHT_B * pred_b_show

    # Isotonic Regressionキャリブレーターは小データで階段関数化し分解能が低下するため使用しない
    # 代わりにレース内正規化で絶対値を補正（馬ごとの相対差は完全に保存される）
    win_sum = race_df["pred_win"].sum()
    show_sum = race_df["pred_show"].sum()
    n_horses = len(race_df)
    if win_sum > 0:
        race_df["pred_win"] = race_df["pred_win"] / win_sum  # 合計→1.0
    if show_sum > 0:
        race_df["pred_show"] = race_df["pred_show"] * (3.0 / show_sum)  # 合計→3.0（3着以内）
    print(f"  ℹ️ レース内正規化適用 — pred_win合計: {win_sum:.2f}→1.00, pred_show合計: {show_sum:.2f}→3.00")

    race_df["pred_finish"] = models["finish"].predict(X_all)

    # Model Bの生予測も保持（市場が見落とすエッジの検出に使用）
    race_df["pred_b_win"] = pred_b_win

    # 期待値
    race_df["ev_win"] = race_df["pred_win"] * race_df["odds"]
    race_df["ev_show"] = race_df["pred_show"] * (race_df["odds"] * 0.3)  # 複勝は概算

    # Kelly fraction（1/4 Kelly）
    race_df["kelly_win"] = race_df.apply(
        lambda row: calc_kelly(row["pred_win"], row["odds"]), axis=1
    )
    race_df["kelly_show"] = race_df.apply(
        lambda row: calc_kelly(row["pred_show"], row["odds"] * 0.3), axis=1
    )

    # △判定用のブレンドshow確率（AI 50% + 市場 50%）
    # LightGBMの分解能不足で同一show_probが多発 → 市場情報で差別化
    _show_odds_approx = (race_df["odds"] * 0.3).clip(lower=1.01)
    _market_show_raw = 1.0 / _show_odds_approx
    _market_show_sum = _market_show_raw.sum()
    if _market_show_sum > 0:
        race_df["_blended_show"] = (
            0.5 * race_df["pred_show"]
            + 0.5 * _market_show_raw * (3.0 / _market_show_sum)
        )
    else:
        race_df["_blended_show"] = race_df["pred_show"]

    # 印（Kelly基準）
    race_df["mark"] = race_df.apply(lambda row: get_mark(row, race_df), axis=1)

    # レーダーチャート（0-100正規化）
    radar_data = {}
    for radar_key, feat_col in RADAR_FEATURES.items():
        if feat_col in race_df.columns:
            normalized = normalize_0_100(race_df[feat_col])
            # instant (best_last3f) は値が小さいほど良いので反転
            if radar_key == "instant":
                normalized = 100 - normalized
            radar_data[radar_key] = normalized
        else:
            radar_data[radar_key] = pd.Series(50, index=race_df.index)

    # JSON構築
    predictions = []
    for _, row in race_df.iterrows():
        radar = {k: int(v.loc[row.name]) if hasattr(v, 'loc') else 50 for k, v in radar_data.items()}

        # last3_results
        last3 = []
        for col in ["last1_finish", "last2_finish", "last3_finish"]:
            val = row.get(col)
            if pd.notna(val):
                last3.append(f"{int(val)}着")
            else:
                last3.append("-")

        predictions.append({
            "horse_number": int(row["horse_number"]),
            "horse_name": str(row["馬名"]),
            "mark": row["mark"],
            "win_prob": round(float(row["pred_win"]), 4),
            "show_prob": round(float(row["pred_show"]), 4),
            "ev_win": round(float(row["ev_win"]), 2),
            "ev_show": round(float(row["ev_show"]), 2),
            "kelly_win": round(float(row["kelly_win"]), 4),
            "kelly_show": round(float(row["kelly_show"]), 4),
            "speed_index": round(float(row["speed_index"]), 1) if pd.notna(row["speed_index"]) else 50.0,
            "radar": radar,
            "comment": generate_comment(row),
            "risk": generate_risk(row),
            "last3_results": last3,
            "odds": {
                "win": round(float(row["odds"]), 1) if pd.notna(row["odds"]) else 10.0,
                "show": round(float(row["odds"]) * 0.3, 1) if pd.notna(row["odds"]) else 3.0,
            },
            "sire": sire_lookup.get(str(row["馬名"]), "不明"),
            "jockey": str(row["騎手"]),
            "frame_number": int(row["frame_number"]),
        })

    # 印順でソート
    mark_order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "×": 4}
    predictions.sort(key=lambda x: (mark_order.get(x["mark"], 99), -x["win_prob"]))

    # === Kelly基準ベースの推奨買い目生成 ===
    BUDGET = 3000  # デフォルト予算
    kelly_horses = [p for p in predictions if p["kelly_win"] > 0]
    top_horses = [p for p in predictions if p["mark"] in ("◎", "○", "▲")]

    bets = []

    # 単勝: Kelly fraction > 0 の馬に比例配分
    for h in kelly_horses:
        amount = max(100, round(BUDGET * h["kelly_win"] / 100) * 100)
        bets.append({
            "type": "単勝",
            "targets": f"{h['horse_number']}番 {h['horse_name']}",
            "amount": amount,
            "ev": h["ev_win"],
            "odds": h["odds"]["win"],
            "kelly": h["kelly_win"],
        })

    # 組み合わせ馬券: 上位馬が2頭以上いれば追加
    if len(top_horses) >= 2:
        targets = "-".join(str(h["horse_number"]) for h in top_horses[:3])
        avg_ev = sum(h["ev_win"] for h in top_horses[:3]) / len(top_horses[:3])
        avg_kelly = sum(h["kelly_win"] for h in top_horses[:3]) / len(top_horses[:3])
        amount = max(100, round(BUDGET * avg_kelly / 100) * 100)
        bets.append({
            "type": "馬連BOX",
            "targets": targets,
            "amount": amount,
            "ev": round(avg_ev, 2),
            "odds": None,
            "kelly": round(avg_kelly, 4),
        })
    if len(top_horses) >= 3:
        top5 = [p for p in predictions if p["mark"] in ("◎", "○", "▲", "△")][:5]
        if len(top5) >= 3:
            targets = "-".join(str(h["horse_number"]) for h in top5)
            avg_ev = sum(h["ev_win"] for h in top5) / len(top5)
            avg_kelly = sum(h["kelly_win"] for h in top5) / len(top5)
            amount = max(100, round(BUDGET * avg_kelly * 0.5 / 100) * 100)
            bets.append({
                "type": "三連複BOX",
                "targets": targets,
                "amount": amount,
                "ev": round(avg_ev, 2),
                "odds": None,
                "kelly": round(avg_kelly, 4),
            })
    if len(top_horses) >= 2:
        h1, h2 = top_horses[0], top_horses[1]
        avg_kelly = (h1["kelly_win"] + h2["kelly_win"]) / 2
        amount = max(100, round(BUDGET * avg_kelly / 100) * 100)
        bets.append({
            "type": "ワイド",
            "targets": f"{h1['horse_number']}-{h2['horse_number']}",
            "amount": amount,
            "ev": round((h1["ev_win"] + h2["ev_win"]) / 2, 2),
            "odds": None,
            "kelly": round(avg_kelly, 4),
        })

    total_inv = sum(b["amount"] for b in bets)
    exp_return = sum(b["amount"] * b["ev"] for b in bets)

    output = {
        "race_info": {
            "name": "第86回桜花賞",
            "date": "2026-04-12",
            "course": "阪神 芝1600m",
            "going": "良",
            "weather": "晴",
            "updated_at": "2026-04-12T10:30:00+09:00",
        },
        "predictions": predictions,
        "recommendations": {
            "headline": "AIが導き出した桜花賞の最強買い目",
            "bets": bets,
            "total_investment": total_inv,
            "expected_return": round(exp_return),
        },
    }

    # 保存
    json_path = BASE_DIR / "frontend" / "src" / "data" / "predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ {json_path}")
    print(f"   出走馬: {len(predictions)}頭")
    print(f"   印: ◎{sum(1 for p in predictions if p['mark']=='◎')} "
          f"○{sum(1 for p in predictions if p['mark']=='○')} "
          f"▲{sum(1 for p in predictions if p['mark']=='▲')} "
          f"△{sum(1 for p in predictions if p['mark']=='△')} "
          f"×{sum(1 for p in predictions if p['mark']=='×')}")


def main() -> None:
    print("=" * 50)
    print("SAKURA ORACLE - モデル学習 & 予測")
    print("=" * 50)

    csv_path = DATA_DIR / "features.csv"
    if not csv_path.exists():
        print("❌ features.csv が見つかりません。先に特徴量生成を実行してください。")
        return

    df = pd.read_csv(csv_path)
    print(f"入力: {len(df)}行 × {len(df.columns)}カラム")

    # 学習
    models = train_models(df)

    # バックテスト
    bt = backtest(df, models)

    # 予測JSON生成
    generate_predictions_json(df, models, bt)

    print("\n完了!")


if __name__ == "__main__":
    main()
