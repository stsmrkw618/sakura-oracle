"""
SAKURA ORACLE — リーケージなし厳密バックテスト

各年の桜花賞を予測する際、「その桜花賞より前のデータのみ」で学習する。
未来データの混入(リーケージ)を完全に排除した上でモデルの期待値を検証。

使い方:
    PYTHONIOENCODING=utf-8 py ml/model/backtest_evaluation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.config import DATA_DIR, BASE_DIR

FEATURE_COLS = [
    "horse_number", "frame_number", "weight", "weight_diff",
    "distance_m", "grade_encoded",
    "total_runs", "win_rate", "show_rate",
    "last1_finish", "last2_finish", "last3_finish",
    "last1_last3f", "last2_last3f", "last1_speed",
    "speed_index", "avg_last3f", "best_last3f",
    "hanshin_runs", "mile_win_rate",
    "jockey_win_rate", "jockey_g1_wins",
    "odds", "popularity",
]

SAKURA_YEARS = [2025, 2024, 2023, 2022, 2021]

PARAMS_BIN = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "verbose": -1,
    "random_state": 42,
}


def extract_year(label: str) -> int:
    """ラベルから年を抽出"""
    import re
    m = re.search(r"(\d{4})", label)
    return int(m.group(1)) if m else 0


def run_strict_backtest(df: pd.DataFrame) -> list[dict]:
    """
    厳密な時系列バックテスト:
    各年の桜花賞を予測する際、その年以前のデータのみで学習する。
    """
    print("=" * 60)
    print("厳密バックテスト（リーケージなし）")
    print("=" * 60)
    print()
    print("方針: 桜花賞Yを予測する際、年 < Y のデータのみで学習")
    print("      （未来データは一切使用しない）")
    print()

    df["year_num"] = df["label"].apply(extract_year)
    df["race_base_name"] = df["label"].apply(
        lambda x: x.replace(str(extract_year(x)), "").replace("(", "").replace(")", "").replace("世代", "")
    )

    results = []

    for target_year in SAKURA_YEARS:
        target_label = f"桜花賞{target_year}"
        test_df = df[df["label"] == target_label].copy()

        if test_df.empty:
            print(f"⚠️  {target_label}: データなし（スキップ）")
            continue

        # ---- 学習データ: target_year より前のレースのみ ----
        train_df = df[
            (df["year_num"] < target_year) &
            (df["label"] != target_label)
        ].copy()

        if len(train_df) < 30:
            print(f"⚠️  {target_label}: 学習データ不足 ({len(train_df)}行)（スキップ）")
            continue

        X_train = train_df[FEATURE_COLS].values
        X_test = test_df[FEATURE_COLS].values

        # --- 勝率モデル ---
        model_win = lgb.LGBMClassifier(**PARAMS_BIN)
        model_win.fit(X_train, train_df["is_win"].values)
        pred_win = model_win.predict_proba(X_test)[:, 1]

        # --- 複勝モデル ---
        model_show = lgb.LGBMClassifier(**PARAMS_BIN)
        model_show.fit(X_train, train_df["is_show"].values)
        pred_show = model_show.predict_proba(X_test)[:, 1]

        test_df["pred_win"] = pred_win
        test_df["pred_show"] = pred_show

        # ---- 評価 ----
        n_horses = len(test_df)

        # 1着予測: モデル最上位 vs 実際の1着
        top1_pred = test_df.nlargest(1, "pred_win")
        actual_winner = test_df[test_df["着順_num"] == 1]
        win_hit = int(top1_pred["馬名"].values[0] in actual_winner["馬名"].values)

        # 3着以内予測: モデル上位3頭 vs 実際3着以内
        top3_pred = test_df.nlargest(3, "pred_show")
        actual_top3 = set(test_df[test_df["着順_num"] <= 3]["馬名"].values)
        pred_top3 = set(top3_pred["馬名"].values)
        show_hits = len(actual_top3 & pred_top3)

        # 1番人気比較（ベースライン）
        fav_horse = test_df[test_df["popularity"] == test_df["popularity"].min()]
        fav_won = int(fav_horse["着順_num"].values[0] <= 1) if len(fav_horse) > 0 else 0
        fav_show = int(fav_horse["着順_num"].values[0] <= 3) if len(fav_horse) > 0 else 0

        # 期待値計算: AI推奨馬の単勝を買った場合
        ai_top1 = top1_pred.iloc[0]
        ai_win_ev = ai_top1["pred_win"] * ai_top1["odds"]
        actual_return_win = float(ai_top1["odds"]) if ai_top1["着順_num"] == 1 else 0.0

        # 期待値計算: AI推奨上位3頭の複勝を均等買い
        show_bets_return = 0.0
        for _, row in top3_pred.iterrows():
            if row["着順_num"] <= 3:
                show_bets_return += row["odds"] * 0.3  # 概算複勝オッズ

        # 結果出力
        winner_name = actual_winner["馬名"].values[0] if len(actual_winner) > 0 else "?"
        pred_name = top1_pred["馬名"].values[0]

        result = {
            "year": target_year,
            "label": target_label,
            "train_size": len(train_df),
            "n_horses": n_horses,
            "ai_pick": pred_name,
            "actual_winner": winner_name,
            "win_hit": win_hit,
            "show_hits": show_hits,
            "fav_won": fav_won,
            "fav_show": fav_show,
            "ai_top3": list(pred_top3),
            "actual_top3": list(actual_top3),
            "ai_win_ev": round(ai_win_ev, 2),
            "actual_return_win": round(actual_return_win, 2),
            "show_return": round(show_bets_return, 2),
            "ai_top1_odds": round(float(ai_top1["odds"]), 1),
            "ai_top1_pop": int(ai_top1["popularity"]) if pd.notna(ai_top1["popularity"]) else 0,
        }
        results.append(result)

        # 詳細表示
        print(f"━━━ {target_label} ({n_horses}頭, 学習: {len(train_df)}行) ━━━")
        print(f"  AI本命: {pred_name} (予測勝率: {ai_top1['pred_win']:.1%}, オッズ: {ai_top1['odds']:.1f}倍)")
        print(f"  実際1着: {winner_name}  →  {'◯ 的中!' if win_hit else '✗ 不的中'}")
        print(f"  AI上位3: {list(pred_top3)}")
        print(f"  実際3着内: {list(actual_top3)}")
        print(f"  複勝的中: {show_hits}/3頭")
        print(f"  1番人気: {'◯勝利' if fav_won else '✗敗退'} / {'◯複勝内' if fav_show else '✗複勝外'}")
        print(f"  期待値(単勝): {ai_win_ev:.2f}  実リターン: {actual_return_win:.1f}倍")
        print()

    return results


def print_summary(results: list[dict]) -> None:
    """サマリー出力"""
    if not results:
        print("結果なし")
        return

    n = len(results)
    print("=" * 60)
    print("サマリー")
    print("=" * 60)

    # 1着的中率
    win_hits = sum(r["win_hit"] for r in results)
    print(f"\n【1着予測】")
    print(f"  的中: {win_hits}/{n} ({win_hits/n:.0%})")
    print(f"  参考: ランダム期待値 = {1/18:.1%} (18頭立て)")

    # 複勝的中率
    total_show = sum(r["show_hits"] for r in results)
    total_possible = n * 3
    print(f"\n【複勝(3着以内)予測】")
    print(f"  的中: {total_show}/{total_possible} ({total_show/total_possible:.0%})")
    print(f"  参考: ランダム期待値 = {3/18:.1%} (18頭中3頭)")

    # 1番人気との比較
    fav_wins = sum(r["fav_won"] for r in results)
    fav_shows = sum(r["fav_show"] for r in results)
    print(f"\n【1番人気ベースライン】")
    print(f"  1着: {fav_wins}/{n} ({fav_wins/n:.0%})")
    print(f"  3着内: {fav_shows}/{n} ({fav_shows/n:.0%})")

    # 単勝回収率 (AI推奨1頭に毎年100円)
    total_invest = n * 100  # 毎年100円
    total_return = sum(r["actual_return_win"] * 100 for r in results)
    roi = total_return / total_invest if total_invest > 0 else 0
    print(f"\n【単勝回収率 (AI本命に毎年100円)】")
    print(f"  投資: {total_invest:,}円 → 回収: {total_return:,.0f}円")
    print(f"  回収率: {roi:.0%}")
    if roi > 1:
        print(f"  → プラス収支! ({total_return - total_invest:,.0f}円の利益)")
    else:
        print(f"  → マイナス収支 ({total_return - total_invest:,.0f}円)")

    # 複勝回収率 (AI上位3頭に均等100円ずつ)
    total_show_invest = n * 3 * 100
    total_show_return = sum(r["show_return"] * 100 for r in results)
    show_roi = total_show_return / total_show_invest if total_show_invest > 0 else 0
    print(f"\n【複勝回収率 (AI上位3頭に各100円)】")
    print(f"  投資: {total_show_invest:,}円 → 回収: {total_show_return:,.0f}円")
    print(f"  回収率: {show_roi:.0%}")

    # 年別一覧テーブル
    print(f"\n{'='*60}")
    print("年別結果一覧")
    print(f"{'='*60}")
    print(f"{'年':>6} {'AI本命':>12} {'1着':>6} {'複勝':>4} {'人気的中':>6} {'EV':>6} {'回収':>6}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['year']:>6} "
            f"{r['ai_pick']:>12} "
            f"{'◯' if r['win_hit'] else '✗':>6} "
            f"{r['show_hits']}/3{'':<1} "
            f"{'◯' if r['fav_won'] else '✗':>6} "
            f"{r['ai_win_ev']:>6.2f} "
            f"{r['actual_return_win']:>5.1f}倍"
        )

    # JSON出力（フロントエンド用）
    output = {
        "summary": {
            "win_rate": round(win_hits / n, 3) if n > 0 else 0,
            "show_rate": round(total_show / total_possible, 3) if total_possible > 0 else 0,
            "win_roi": round(roi, 3),
            "show_roi": round(show_roi, 3),
            "fav_win_rate": round(fav_wins / n, 3) if n > 0 else 0,
            "random_win_rate": round(1 / 18, 3),
            "random_show_rate": round(3 / 18, 3),
            "years_tested": n,
        },
        "yearly": [
            {
                "year": r["year"],
                "ai_pick": r["ai_pick"],
                "actual_winner": r["actual_winner"],
                "win_hit": r["win_hit"],
                "show_hits": r["show_hits"],
                "ai_top3": r["ai_top3"],
                "actual_top3": r["actual_top3"],
                "ai_win_ev": r["ai_win_ev"],
                "actual_return": r["actual_return_win"],
                "fav_won": r["fav_won"],
            }
            for r in results
        ],
    }

    json_path = BASE_DIR / "frontend" / "src" / "data" / "backtest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ backtest.json 保存: {json_path}")


def main() -> None:
    csv_path = DATA_DIR / "features.csv"
    if not csv_path.exists():
        print("❌ features.csv が見つかりません")
        return

    df = pd.read_csv(csv_path)
    print(f"入力: {len(df)}行 × {len(df.columns)}カラム\n")

    results = run_strict_backtest(df)
    print_summary(results)

    print("\n完了!")


if __name__ == "__main__":
    main()
