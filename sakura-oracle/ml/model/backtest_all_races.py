"""
SAKURA ORACLE — 全重賞Walk-Forwardバックテスト

全レースに対してリーケージなしの時系列バックテストを実行。
各レースを予測する際、「そのレースより前のデータのみ」で学習する。

使い方:
    PYTHONIOENCODING=utf-8 py ml/model/backtest_all_races.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

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


def run_walk_forward(df: pd.DataFrame) -> list[dict]:
    """Walk-Forwardバックテスト（デュアルモデル + 正則化 + 早期停止）"""
    print("=" * 60)
    print("全重賞 Walk-Forward バックテスト（改良版）")
    print("=" * 60)
    print()
    print("方針: 各レースRを予測する際、race_id < R のデータのみで学習")
    print(f"モデル: A(市場連動={BLEND_WEIGHT_A}) + B(エッジ検出={BLEND_WEIGHT_B})")
    print()

    races = build_race_order(df)
    results = []
    min_train_size = 50

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
            "quinella_box3_hit": quinella_box3_hit,
            "wide_top2_hit": wide_top2_hit,
            "trio_box3_hit": trio_box3_hit,
            "trio_box5_hit": trio_box5_hit,
        })

    return results


def print_summary(results: list[dict]) -> None:
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

    print(f"\n平均出走頭数: {avg_horses:.1f}頭")
    print(f"\n{'':>20} {'AI':>10} {'1番人気':>10} {'ランダム':>10}")
    print("-" * 55)
    print(f"{'1着的中率':>20} {win_hits/n:>9.1%} {fav_wins/n:>9.1%} {1/avg_horses:>9.1%}")
    print(f"{'複勝的中率':>20} {show_hits_total/show_possible:>9.1%} {fav_shows/n:>9.1%} {3/avg_horses:>9.1%}")
    print(f"{'単勝回収率':>20} {win_roi:>9.0%} {'---':>10} {'---':>10}")
    print(f"{'複勝回収率':>20} {show_roi:>9.0%} {'---':>10} {'---':>10}")

    print(f"\n{'組合せ馬券的中率':>20}")
    print(f"  馬連BOX(3): {quinella_hits}/{n} ({quinella_hits/n:.0%})")
    print(f"  ワイド(◎-○): {wide_hits}/{n} ({wide_hits/n:.0%})")
    print(f"  三連複BOX(3): {trio3_hits}/{n} ({trio3_hits/n:.0%})")
    print(f"  三連複BOX(5): {trio5_hits}/{n} ({trio5_hits/n:.0%})")

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
        },
        "combo_hit_rates": {
            "quinella_box3": round(quinella_hits / n, 3) if n > 0 else 0,
            "wide_top2": round(wide_hits / n, 3) if n > 0 else 0,
            "trio_box3": round(trio3_hits / n, 3) if n > 0 else 0,
            "trio_box5": round(trio5_hits / n, 3) if n > 0 else 0,
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
    results = run_walk_forward(df)
    print_summary(results)
    print("\n完了!")


if __name__ == "__main__":
    main()
