"""
SAKURA ORACLE — 任意レース予測CLI

任意の3歳牝馬重賞レースに対して予測を行い、predictions.json を生成する。

使い方:
    PYTHONIOENCODING=utf-8 py ml/model/predict_race.py チューリップ賞2026
    PYTHONIOENCODING=utf-8 py ml/model/predict_race.py フェアリーS2026

処理フロー:
    1. 引数からレース名パース → RACE_META で距離/会場/グレード取得
    2. TARGET_RACES から race_id 特定
    3. 出馬表スクレイプ → 各馬の特徴量を features.csv からルックアップ
    4. Walk-Forward モデル学習 → デュアルモデル予測 → predictions.json 生成
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.scraper.config import TARGET_RACES, DATA_DIR, BASE_DIR
from ml.scraper.race_scraper import find_race_id_from_date
from ml.scraper.entry_scraper import scrape_entries
from ml.scraper.horse_history_scraper import build_features_from_history
from ml.model.feature_engineering import RACE_META, get_race_base
from ml.model.backtest_all_races import (
    _make_params_bin,
    _train_model,
    _get_available_features,
    FEATURE_COLS_ALL,
    FEATURE_COLS_NO_ODDS,
    BLEND_WEIGHT_A,
    BLEND_WEIGHT_B,
)
from ml.model.predictor import (
    calc_kelly,
    get_mark,
    generate_comment,
    generate_risk,
    normalize_0_100,
    RADAR_FEATURES,
)


RACE_SLUG: dict[str, str] = {
    "桜花賞": "sakura",
    "チューリップ賞": "tulip",
    "フィリーズレビュー": "fillies",
    "クイーンC": "queenc",
    "フェアリーS": "fairy",
    "阪神JF": "hanshinjf",
    "アルテミスS": "artemis",
    "ファンタジーS": "fantasy",
    "オークス": "oaks",
    "秋華賞": "shuka",
    "ローズS": "rose",
    "紫苑S": "shion",
}


def _parse_race_arg(arg: str) -> tuple[str, str]:
    """コマンドライン引数からレース基本名と年を抽出する。

    Args:
        arg: "チューリップ賞2026" や "フェアリーS2026" など

    Returns:
        (race_base, year): ("チューリップ賞", "2026")
    """
    m = re.search(r"(\d{4})", arg)
    year = m.group(1) if m else ""
    # 年部分を除去してレース名を得る
    name_part = re.sub(r"\d{4}", "", arg).strip()

    # RACE_META のキーとマッチ
    for key in RACE_META:
        if key in name_part or name_part in key:
            return key, year

    # フォールバック: TARGET_RACES の keyword でマッチ
    for race in TARGET_RACES:
        if name_part in race["keyword"] or race["keyword"] in name_part:
            race_base = get_race_base(race["label"])
            return race_base, year

    return name_part, year


def _find_target_race(race_base: str, year: str) -> dict | None:
    """TARGET_RACES からレース基本名+年で一致するエントリを検索する。"""
    for race in TARGET_RACES:
        if race_base in race["keyword"] or race["keyword"] in race_base:
            if year in race["label"]:
                return race
    # ラベルに完全一致も試行
    for race in TARGET_RACES:
        if race_base in race["label"] and year in race["label"]:
            return race
    return None


def _parse_weight_str(weight_str: str) -> tuple[float | None, float | None]:
    """馬体重文字列 '480(+4)' → (480.0, 4.0)"""
    if pd.isna(weight_str):
        return None, None
    s = str(weight_str).strip()
    m = re.match(r"(\d+)\(([+-]?\d+)\)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.match(r"(\d+)", s)
    if m2:
        return float(m2.group(1)), None
    return None, None


def _build_prediction_features(
    entries: pd.DataFrame,
    features_df: pd.DataFrame,
    race_meta: dict,
    race_id: str,
    race_date: str = "",
) -> pd.DataFrame:
    """出馬表と features.csv を結合して予測用特徴量を構築する。

    Args:
        entries: 出馬表 DataFrame (枠番, 馬番, 馬名, 騎手, 単勝オッズ, 人気, ...)
        features_df: features.csv 全体
        race_meta: RACE_META から取得した {"distance", "venue", "grade", ...}
        race_id: 対象レースの race_id
        race_date: レース開催日 (YYYYMMDD) — race_info.csvにない新規レース用フォールバック

    Returns:
        予測用 DataFrame（FEATURE_COLS_ALL のカラムを持つ）
    """
    rows = []
    missing_horses: list[str] = []

    # 日付ルックアップ構築（出走間隔計算用）
    race_info_path = DATA_DIR / "raw" / "race_info.csv"
    date_lookup: dict[str, str] = {}
    if race_info_path.exists():
        ri = pd.read_csv(race_info_path, dtype={"race_id": str, "date": str})
        date_lookup = dict(zip(ri["race_id"], ri["date"]))
    # race_info.csvにない場合はrace_dateをフォールバックで使用
    date_str = date_lookup.get(race_id) or race_date
    target_date = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce") if date_str else None

    # features.csv の馬名ベースで最新行を取得
    # race_id < 対象レース のデータのみ使用（リーケージ防止）
    hist_df = features_df[features_df["race_id"].astype(str) < race_id].copy()

    # 馬名ごとに最新行をルックアップ
    latest_by_horse: dict[str, pd.Series] = {}
    if not hist_df.empty:
        hist_df = hist_df.sort_values("race_id")
        for name, group in hist_df.groupby("馬名"):
            latest_by_horse[str(name)] = group.iloc[-1]

    # 騎手ごとの最新成績
    latest_jockey: dict[str, pd.Series] = {}
    if not hist_df.empty:
        for jockey, group in hist_df.groupby("騎手"):
            latest_jockey[str(jockey)] = group.iloc[-1]

    # field_strength 計算用: 全頭のオッズ逆数合計
    raw_odds = entries.get("単勝オッズ")
    if raw_odds is not None:
        odds_series = pd.to_numeric(raw_odds, errors="coerce")
        # 全NaN（オッズ未公開）の場合はデフォルト値を使用
        if odds_series.notna().sum() == 0:
            print("  ⚠️ オッズ未公開 — 均等オッズ(10.0)で仮計算")
            odds_series = pd.Series([10.0] * len(entries))
        else:
            odds_series = odds_series.clip(lower=1.0)
    else:
        print("  ⚠️ オッズカラムなし — 均等オッズ(10.0)で仮計算")
        odds_series = pd.Series([10.0] * len(entries))
    odds_inv_sum = (1.0 / odds_series).sum()

    # 対象レース日の文字列（horse_history_scraper用）
    target_date_str = date_lookup.get(race_id) or race_date

    # 馬ページスクレイピングで補完された馬のリスト
    scraped_horses: list[str] = []
    # 全フォールバック（中央値）の馬のリスト
    median_horses: list[str] = []

    for _, entry in entries.iterrows():
        horse_name = str(entry["馬名"])
        hist = latest_by_horse.get(horse_name)

        if hist is not None:
            # 過去データあり → features.csv から引き継ぎ
            row = hist.to_dict()
        else:
            # features.csv にない → 馬ページから全戦績を取得して特徴量構築
            horse_id = str(entry.get("horse_id", ""))
            scraped_features = None
            if horse_id:
                scraped_features = build_features_from_history(
                    horse_id,
                    target_date=target_date_str,
                    horse_name=horse_name,
                )

            if scraped_features is not None:
                # 馬ページから特徴量構築成功
                row = scraped_features
                scraped_horses.append(horse_name)
            else:
                # スクレイピング失敗 → 中央値フォールバック
                missing_horses.append(horse_name)
                median_horses.append(horse_name)
                row = {}
                for col in FEATURE_COLS_ALL:
                    if col in hist_df.columns:
                        row[col] = hist_df[col].median()
                    else:
                        row[col] = 0

        # 出馬表から上書き
        row["馬名"] = horse_name
        row["horse_number"] = int(entry["馬番"])
        row["frame_number"] = int(entry.get("枠番", 0))
        row["distance_m"] = race_meta.get("distance", 1600)
        row["grade_encoded"] = race_meta.get("grade", 3)
        row["race_id"] = race_id

        # オッズ
        odds_val = pd.to_numeric(entry.get("単勝オッズ"), errors="coerce")
        row["odds"] = float(odds_val) if pd.notna(odds_val) else 10.0
        pop_val = pd.to_numeric(entry.get("人気"), errors="coerce")
        row["popularity"] = float(pop_val) if pd.notna(pop_val) else 9.0

        # field_strength
        if pd.notna(odds_val) and odds_val > 0:
            row["field_strength"] = (1.0 / max(float(odds_val), 1.0)) / max(odds_inv_sum, 0.01)
        else:
            row["field_strength"] = 1.0 / len(entries)

        # 斤量 → weight（スクレイピング特徴量より出馬表の斤量を優先）
        weight_val = entry.get("斤量")
        if pd.notna(weight_val):
            row["weight"] = float(weight_val)

        # 馬体重（出馬表にある場合）
        if "馬体重" in entry.index and pd.notna(entry["馬体重"]):
            w, wd = _parse_weight_str(str(entry["馬体重"]))
            if w is not None:
                old_weight = row.get("weight")
                row["weight"] = w
                if wd is not None:
                    row["weight_diff"] = wd
                elif old_weight and old_weight > 0:
                    row["weight_diff"] = w - old_weight

        # 出走間隔（features.csv経由の馬のみ従来ロジック、スクレイピング馬はrest_weeks設定済み）
        if hist is not None and target_date is not None:
            last_race_id = str(hist.get("race_id", ""))
            last_date_str = date_lookup.get(last_race_id)
            if last_date_str:
                last_date = pd.to_datetime(last_date_str, format="%Y%m%d")
                row["rest_weeks"] = (target_date - last_date).days / 7.0

        # 騎手情報
        jockey_name = str(entry.get("騎手", ""))
        row["騎手"] = jockey_name
        jockey_hist = latest_jockey.get(jockey_name)
        if jockey_hist is not None:
            row["jockey_win_rate"] = jockey_hist.get("jockey_win_rate", 0)
            row["jockey_g1_wins"] = jockey_hist.get("jockey_g1_wins", 0)

        rows.append(row)

    # 結果サマリー表示
    if scraped_horses:
        print(f"\n  ✅ 馬ページから特徴量補完: {len(scraped_horses)}頭")
        for h in scraped_horses:
            print(f"    ○ {h}")
    if median_horses:
        print(f"\n  ⚠️ 中央値フォールバック: {len(median_horses)}頭")
        for h in median_horses:
            print(f"    × {h}")

    pred_df = pd.DataFrame(rows)

    # 欠損値補完
    for col in FEATURE_COLS_ALL:
        if col in pred_df.columns and pred_df[col].isnull().any():
            median_val = hist_df[col].median() if col in hist_df.columns else 0
            pred_df[col] = pred_df[col].fillna(median_val)

    return pred_df


def predict_race(race_label: str, override_race_id: str | None = None) -> None:
    """メイン予測処理。

    Args:
        race_label: "チューリップ賞2026" のようなレースラベル
        override_race_id: race_idを直接指定（netkeibaで自動検出できない場合）
    """
    print("=" * 60)
    print(f"SAKURA ORACLE — {race_label} 予測")
    print("=" * 60)

    # --- 1. レース情報パース ---
    race_base, year = _parse_race_arg(race_label)
    meta = RACE_META.get(race_base)
    if meta is None:
        print(f"ERROR: RACE_META に '{race_base}' が見つかりません")
        print(f"  利用可能: {list(RACE_META.keys())}")
        sys.exit(1)

    venue = meta["venue"]
    distance = meta["distance"]
    grade = meta["grade"]
    surface = meta["surface"]

    print(f"  レース: {race_base} ({year})")
    print(f"  コース: {venue} {surface}{distance}m")
    print(f"  グレード: G{1 if grade == 5 else 2 if grade == 3 else 3}")

    # --- 2. race_id 特定 ---
    target = _find_target_race(race_base, year)
    if override_race_id:
        race_id = override_race_id
        print(f"  race_id: {race_id}（手動指定）")
    else:
        if target is None:
            print(f"ERROR: TARGET_RACES に '{race_base}{year}' が見つかりません")
            sys.exit(1)

        race_id = find_race_id_from_date(target["date"], target["keyword"], label=target["label"])
        if race_id is None:
            print(f"ERROR: race_id 特定失敗 (date={target['date']}, keyword={target['keyword']})")
            print(f"  ヒント: --race-id オプションで直接指定できます")
            print(f"  例: py ml/model/predict_race.py {race_label} --race-id 202609010411")
            sys.exit(1)

    print(f"  race_id: {race_id}")

    # --- 3. 出馬表取得 ---
    print("\n--- 出馬表取得 ---")
    entries = scrape_entries(race_id)
    print(f"  {len(entries)}頭の出走馬を取得")

    # --- 4. features.csv ロード & 特徴量構築 ---
    print("\n--- 特徴量構築 ---")
    csv_path = DATA_DIR / "features.csv"
    if not csv_path.exists():
        print("ERROR: features.csv が見つかりません。先に特徴量生成を実行してください。")
        sys.exit(1)

    features_df = pd.read_csv(csv_path)
    print(f"  features.csv: {len(features_df)}行ロード")

    # target["date"] はrace_info.csvにない新規レース用のフォールバック日付
    race_date = target["date"] if target else ""
    pred_df = _build_prediction_features(entries, features_df, meta, race_id, race_date=race_date)
    print(f"  予測用データ: {len(pred_df)}頭 × {len(pred_df.columns)}カラム")

    # --- 5. Walk-Forward モデル学習 ---
    print("\n--- モデル学習 ---")
    train_df = features_df[features_df["race_id"].astype(str) < race_id].copy()
    print(f"  訓練データ: {len(train_df)}行 (race_id < {race_id})")

    if len(train_df) < 50:
        print("WARNING: 訓練データが少ないです（50行未満）。予測精度に影響する可能性があります。")

    feat_all = _get_available_features(train_df, FEATURE_COLS_ALL)
    feat_no_odds = _get_available_features(train_df, FEATURE_COLS_NO_ODDS)

    X_train_all = train_df[feat_all].values
    X_train_no_odds = train_df[feat_no_odds].values
    y_win = train_df["is_win"].values
    y_show = train_df["is_show"].values

    # Optuna最適化済み scale_pos_weight
    params_win = _make_params_bin(scale_pos_weight=11.817)
    params_show = _make_params_bin(scale_pos_weight=6.988)

    # Model A: 全特徴量（市場連動型）
    model_a_win = _train_model(X_train_all, y_win, params_win)
    model_a_show = _train_model(X_train_all, y_show, params_show)

    # Model B: オッズ除外（エッジ検出型）
    model_b_win = _train_model(X_train_no_odds, y_win, params_win)
    model_b_show = _train_model(X_train_no_odds, y_show, params_show)

    print(f"  Model A 特徴量: {len(feat_all)}個")
    print(f"  Model B 特徴量: {len(feat_no_odds)}個")

    # --- 6. 予測 ---
    print("\n--- 予測実行 ---")

    # 予測用の特徴量を確保（欠損カラムは0で埋める）
    for col in feat_all:
        if col not in pred_df.columns:
            pred_df[col] = 0
    for col in feat_no_odds:
        if col not in pred_df.columns:
            pred_df[col] = 0

    X_pred_all = pred_df[feat_all].values
    X_pred_no_odds = pred_df[feat_no_odds].values

    pred_a_win = model_a_win.predict_proba(X_pred_all)[:, 1]
    pred_b_win = model_b_win.predict_proba(X_pred_no_odds)[:, 1]
    pred_df["pred_win"] = BLEND_WEIGHT_A * pred_a_win + BLEND_WEIGHT_B * pred_b_win

    pred_a_show = model_a_show.predict_proba(X_pred_all)[:, 1]
    pred_b_show = model_b_show.predict_proba(X_pred_no_odds)[:, 1]
    pred_df["pred_show"] = BLEND_WEIGHT_A * pred_a_show + BLEND_WEIGHT_B * pred_b_show

    # レース内正規化（predictor.pyと同一ロジック）
    # Isotonic Regressionは小データで階段関数化し分解能が低下するため使用しない
    # 代わりにレース内正規化で絶対値を補正（馬ごとの相対差は完全に保存される）
    win_sum = pred_df["pred_win"].sum()
    show_sum = pred_df["pred_show"].sum()
    if win_sum > 0:
        pred_df["pred_win"] = pred_df["pred_win"] / win_sum  # 合計→1.0
    if show_sum > 0:
        pred_df["pred_show"] = pred_df["pred_show"] * (3.0 / show_sum)  # 合計→3.0
    print(f"  ✅ レース内正規化適用済み（win合計=1.0, show合計=3.0）")

    pred_df["pred_b_win"] = pred_b_win

    # 期待値
    pred_df["ev_win"] = pred_df["pred_win"] * pred_df["odds"]
    pred_df["ev_show"] = pred_df["pred_show"] * (pred_df["odds"] * 0.3)

    # Kelly fraction
    pred_df["kelly_win"] = pred_df.apply(
        lambda row: calc_kelly(row["pred_win"], row["odds"]), axis=1
    )
    pred_df["kelly_show"] = pred_df.apply(
        lambda row: calc_kelly(row["pred_show"], row["odds"] * 0.3), axis=1
    )

    # 印
    pred_df["mark"] = pred_df.apply(lambda row: get_mark(row, pred_df), axis=1)

    # --- 7. predictions.json 生成 ---
    print("\n--- predictions.json 生成 ---")

    # 血統データ読み込み（sire表示用）
    pedigree_path = DATA_DIR / "raw" / "horse_pedigree.csv"
    sire_lookup: dict[str, str] = {}
    if pedigree_path.exists():
        ped_df = pd.read_csv(pedigree_path)
        sire_lookup = dict(zip(ped_df["horse_name"], ped_df["sire"]))

    # レーダーチャート
    radar_data: dict[str, pd.Series] = {}
    for radar_key, feat_col in RADAR_FEATURES.items():
        if feat_col in pred_df.columns:
            normalized = normalize_0_100(pred_df[feat_col])
            if radar_key == "instant":
                normalized = 100 - normalized
            radar_data[radar_key] = normalized
        else:
            radar_data[radar_key] = pd.Series(50, index=pred_df.index)

    predictions = []
    for _, row in pred_df.iterrows():
        radar = {
            k: int(v.loc[row.name]) if hasattr(v, "loc") else 50
            for k, v in radar_data.items()
        }

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
            "speed_index": round(float(row.get("speed_index", 50)), 1),
            "radar": radar,
            "comment": generate_comment(row),
            "risk": generate_risk(row, venue=str(venue)),
            "last3_results": last3,
            "odds": {
                "win": round(float(row["odds"]), 1),
                "show": round(float(row["odds"]) * 0.3, 1),
            },
            "sire": sire_lookup.get(str(row["馬名"]), "不明"),
            "jockey": str(row.get("騎手", "不明")),
            "frame_number": int(row["frame_number"]),
        })

    # 印順でソート
    mark_order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "×": 4}
    predictions.sort(key=lambda x: (mark_order.get(x["mark"], 99), -x["win_prob"]))

    # === Kelly基準ベースの推奨買い目生成 ===
    BUDGET = 3000
    kelly_horses = [p for p in predictions if p["kelly_win"] > 0]
    top_horses = [p for p in predictions if p["mark"] in ("◎", "○", "▲")]

    bets = []

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

    # 日付フォーマット
    date_str = target["date"]  # "20260301"
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    # グレード表記
    grade_label = {5: "G1", 3: "G2"}.get(grade, "G3")

    output = {
        "race_info": {
            "name": f"{race_base}({grade_label})",
            "date": formatted_date,
            "course": f"{venue} {surface}{distance}m",
            "going": "TBD",
            "weather": "TBD",
            "updated_at": datetime.now().isoformat(),
        },
        "predictions": predictions,
        "recommendations": {
            "headline": f"AIが導き出した{race_base}の最強買い目",
            "bets": bets,
            "total_investment": total_inv,
            "expected_return": round(exp_return),
        },
    }

    # predictions.jsonは桜花賞（メインレース）専用 — 他レースは上書きしない
    if race_base == "桜花賞":
        json_path = BASE_DIR / "frontend" / "src" / "data" / "predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"  predictions.json 更新（メインレース）")

    # --- public/races/ 出力（フロントエンドのレース選択UI用）---
    slug = RACE_SLUG.get(race_base, race_base)
    race_file_id = f"{slug}{year}"
    races_dir = BASE_DIR / "frontend" / "public" / "races"
    races_dir.mkdir(parents=True, exist_ok=True)

    race_json_path = races_dir / f"{race_file_id}.json"
    with open(race_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # index.json upsert
    index_path = races_dir / "index.json"
    index_data: list[dict] = []
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            index_data = []

    entry = {
        "id": race_file_id,
        "label": f"{race_base}({grade_label}) {year}",
        "date": formatted_date,
        "course": f"{venue} {surface}{distance}m",
    }
    # upsert by id
    index_data = [e for e in index_data if e.get("id") != race_file_id]
    index_data.append(entry)
    # sort by date descending
    index_data.sort(key=lambda e: e.get("date", ""), reverse=True)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    print(f"  races/{race_file_id}.json 保存: {race_json_path}")
    print(f"  races/index.json 更新: {len(index_data)}レース")

    # --- 結果表示 ---
    print(f"\n{'='*60}")
    print(f"  {race_base} {year} 予測結果")
    print(f"{'='*60}")
    print(f"  {'馬番':>4} {'馬名':>10} {'印':>2} {'勝率':>6} {'複勝率':>6} {'EV':>5} {'Kelly':>7}")
    print("-" * 60)
    for p in predictions:
        print(
            f"  {p['horse_number']:>4} {p['horse_name']:>10} "
            f"{p['mark']:>2} {p['win_prob']:>5.1%} {p['show_prob']:>5.1%} "
            f"{p['ev_win']:>5.2f} {p['kelly_win']:>7.4f}"
        )

    print(f"\n  races/{race_file_id}.json 保存完了")
    print(f"  出走馬: {len(predictions)}頭")
    marks = {m: sum(1 for p in predictions if p["mark"] == m) for m in ["◎", "○", "▲", "△", "×"]}
    print(f"  印: ◎{marks['◎']} ○{marks['○']} ▲{marks['▲']} △{marks['△']} ×{marks['×']}")
    print("\n完了!")


def main() -> None:
    if len(sys.argv) < 2:
        print("使い方: PYTHONIOENCODING=utf-8 py ml/model/predict_race.py <レース名+年> [--race-id <id>]")
        print()
        print("例:")
        print("  py ml/model/predict_race.py チューリップ賞2026")
        print("  py ml/model/predict_race.py チューリップ賞2026 --race-id 202609010411")
        print("  py ml/model/predict_race.py フィリーズレビュー2026")
        print("  py ml/model/predict_race.py フェアリーS2026")
        print("  py ml/model/predict_race.py 桜花賞2025")
        print()
        print("利用可能なレース:")
        for key in RACE_META:
            print(f"  - {key}")
        sys.exit(1)

    race_label = sys.argv[1]
    # --race-id オプション解析
    override_race_id = None
    if "--race-id" in sys.argv:
        idx = sys.argv.index("--race-id")
        if idx + 1 < len(sys.argv):
            override_race_id = sys.argv[idx + 1]
    predict_race(race_label, override_race_id=override_race_id)


if __name__ == "__main__":
    main()
