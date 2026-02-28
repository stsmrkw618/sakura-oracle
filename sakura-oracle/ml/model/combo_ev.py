"""
SAKURA ORACLE — ExcelオッズからAI推奨TOP10買い目を生成

netkeibaから取得したExcel（シート別に馬連・ワイド・三連複・馬単・三連単のTOP50オッズ）を
読み込み、AI確率×実オッズでEVを算出し、上位10件をJSONに出力する。

使い方:
    PYTHONIOENCODING=utf-8 py ml/model/combo_ev.py --excel チューリップ.xlsx --race tulip2026
"""

import json
import re
import sys
from itertools import permutations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.config import BASE_DIR


# --- シート名→馬券タイプのマッピング ---
SHEET_MAP: dict[str, str] = {
    "馬連": "馬連",
    "ワイド": "ワイド",
    "三連複": "三連複",
    "馬単": "馬単",
    "三連単": "三連単",
}


def _parse_wide_odds(val: str | float) -> float:
    """ワイドの "2.0  2.2" レンジ表記を平均値に変換する。"""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # "2.0 - 2.2" or "2.0  2.2" or "2.0-2.2" のようなレンジ
    parts = re.split(r"[\s\-~～]+", s)
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            continue
    if len(nums) >= 2:
        return sum(nums) / len(nums)
    elif len(nums) == 1:
        return nums[0]
    return float("nan")


def _parse_horse_numbers(val: str | float) -> list[int]:
    """馬番文字列をintのリストに変換する。

    "3 - 12" → [3, 12]
    "1 - 5 - 9" → [1, 5, 9]
    """
    s = str(val).strip()
    parts = re.split(r"[\s\-→]+", s)
    nums = []
    for p in parts:
        p = p.strip()
        if p.isdigit():
            nums.append(int(p))
    return nums


def _load_excel_odds(excel_path: str | Path) -> list[dict]:
    """Excelファイルから全シートの組合せオッズを読み込む。

    Returns:
        [{"type": "馬連", "horses": [3, 12], "odds": 201.1}, ...]
    """
    xls = pd.ExcelFile(excel_path)
    results = []

    for sheet_name in xls.sheet_names:
        # シート名からタイプを判定
        bet_type = None
        for key, val in SHEET_MAP.items():
            if key in sheet_name:
                bet_type = val
                break
        if bet_type is None:
            print(f"  ⚠️ 不明なシート名: {sheet_name}（スキップ）")
            continue

        df = xls.parse(sheet_name)
        if df.empty:
            continue

        # カラム名の正規化（最初の2列が馬番組合せ+オッズの想定）
        # netkeibaのExcelは「組合せ」「オッズ」のような列名
        cols = list(df.columns)

        # 組合せ列とオッズ列を特定
        combo_col = None
        odds_col = None
        for c in cols:
            c_str = str(c).strip()
            if "組" in c_str or "番" in c_str or "馬" in c_str:
                combo_col = c
            elif "オッズ" in c_str or "odds" in c_str.lower() or "倍" in c_str:
                odds_col = c

        # カラム名で見つからない場合は位置ベースで推定
        if combo_col is None and len(cols) >= 1:
            combo_col = cols[0]
        if odds_col is None and len(cols) >= 2:
            odds_col = cols[1]

        if combo_col is None or odds_col is None:
            print(f"  ⚠️ {sheet_name}: カラム特定失敗（スキップ）")
            continue

        for _, row in df.iterrows():
            combo_val = row[combo_col]
            odds_val = row[odds_col]

            if pd.isna(combo_val) or pd.isna(odds_val):
                continue

            horses = _parse_horse_numbers(str(combo_val))
            if not horses:
                continue

            # オッズ変換（ワイドはレンジ表記対応）
            if bet_type == "ワイド":
                odds = _parse_wide_odds(odds_val)
            else:
                try:
                    odds = float(odds_val)
                except (ValueError, TypeError):
                    continue

            if pd.isna(odds) or odds <= 0:
                continue

            results.append({
                "type": bet_type,
                "horses": horses,
                "odds": odds,
            })

    print(f"  Excel読込: {len(results)}件のオッズデータ")
    return results


# --- Harville確率エンジン ---

def _harville_exacta(probs: dict[int, float], first: int, second: int) -> float:
    """Harville: P(first 1着, second 2着)"""
    p_a = probs.get(first, 0)
    p_b = probs.get(second, 0)
    denom = 1.0 - p_a
    if denom <= 0:
        return 0.0
    return p_a * (p_b / denom)


def _harville_trifecta(probs: dict[int, float], a: int, b: int, c: int) -> float:
    """Harville: P(a 1着, b 2着, c 3着)"""
    p_a = probs.get(a, 0)
    p_b = probs.get(b, 0)
    p_c = probs.get(c, 0)
    d1 = 1.0 - p_a
    d2 = 1.0 - p_a - p_b
    if d1 <= 0 or d2 <= 0:
        return 0.0
    return p_a * (p_b / d1) * (p_c / d2)


def _combo_prob(
    probs: dict[int, float],
    bet_type: str,
    horses: list[int],
) -> float:
    """馬券タイプに応じた組合せ確率を算出する。

    - 馬連(quinella): A,Bが1-2着（順不問）
    - ワイド(wide): A,Bが両方3着以内
    - 三連複(trio): A,B,Cが1-2-3着（順不問）
    - 馬単(exacta): A→B（着順固定）
    - 三連単(trifecta): A→B→C（着順固定）
    """
    if bet_type == "馬連":
        if len(horses) < 2:
            return 0.0
        a, b = horses[0], horses[1]
        return _harville_exacta(probs, a, b) + _harville_exacta(probs, b, a)

    elif bet_type == "ワイド":
        if len(horses) < 2:
            return 0.0
        a, b = horses[0], horses[1]
        # P(A,B両方3着以内) = Σ_k trio(A,B,k)
        total = 0.0
        for k in probs:
            if k == a or k == b:
                continue
            total += _harville_trio_all_perms(probs, a, b, k)
        return total

    elif bet_type == "三連複":
        if len(horses) < 3:
            return 0.0
        return _harville_trio_all_perms(probs, horses[0], horses[1], horses[2])

    elif bet_type == "馬単":
        if len(horses) < 2:
            return 0.0
        return _harville_exacta(probs, horses[0], horses[1])

    elif bet_type == "三連単":
        if len(horses) < 3:
            return 0.0
        return _harville_trifecta(probs, horses[0], horses[1], horses[2])

    return 0.0


def _harville_trio_all_perms(probs: dict[int, float], a: int, b: int, c: int) -> float:
    """三連複: 6通りの順列合計"""
    total = 0.0
    for perm in permutations([a, b, c]):
        total += _harville_trifecta(probs, perm[0], perm[1], perm[2])
    return total


def _calc_kelly(prob: float, odds: float, fraction: float = 0.25) -> float:
    """1/4 Kelly基準で最適賭け比率を計算する。"""
    if odds <= 1.0 or prob <= 0 or prob >= 1:
        return 0.0
    b = odds - 1.0  # ネットオッズ
    f = (prob * b - (1 - prob)) / b
    return max(0.0, f * fraction)


def generate_top_bets(
    excel_path: str | Path,
    race_json_path: str | Path,
    output_path: Optional[str | Path] = None,
    top_n: int = 10,
) -> list[dict]:
    """ExcelオッズとAI確率からEV上位の買い目を生成する。

    Args:
        excel_path: netkeibaオッズExcelのパス
        race_json_path: races/{slug}{year}.json のパス
        output_path: 出力先パス（Noneの場合はrace_json_pathの横に _top_bets.json）
        top_n: 上位何件を出力するか

    Returns:
        上位N件の買い目リスト
    """
    # --- AI確率を読み込む ---
    with open(race_json_path, "r", encoding="utf-8") as f:
        race_data = json.load(f)

    predictions = race_data["predictions"]
    # 馬番→馬名のマッピング
    name_map: dict[int, str] = {}
    raw_probs: dict[int, float] = {}
    for p in predictions:
        num = p["horse_number"]
        name_map[num] = p["horse_name"]
        raw_probs[num] = p["win_prob"]

    # 正規化（合計=1.0にする）
    total_prob = sum(raw_probs.values())
    norm_probs: dict[int, float] = {}
    if total_prob > 0:
        for k, v in raw_probs.items():
            norm_probs[k] = v / total_prob
    else:
        norm_probs = raw_probs

    print(f"  AI確率: {len(norm_probs)}頭ロード済み")

    # --- Excelオッズ読込 ---
    all_odds = _load_excel_odds(excel_path)

    if not all_odds:
        print("  ⚠️ オッズデータが空です")
        return []

    # --- EV算出 ---
    results = []
    for entry in all_odds:
        bet_type = entry["type"]
        horses = entry["horses"]
        odds = entry["odds"]

        # 馬番がAIデータに存在するか確認
        valid = all(h in norm_probs for h in horses)
        if not valid:
            continue

        prob = _combo_prob(norm_probs, bet_type, horses)
        if prob <= 0:
            continue

        ev = prob * odds
        kelly = _calc_kelly(prob, odds)

        # 馬名の組合せ文字列
        names = "-".join(name_map.get(h, str(h)) for h in horses)
        targets = "-".join(str(h) for h in horses)

        results.append({
            "type": bet_type,
            "targets": targets,
            "names": names,
            "odds": round(odds, 1),
            "prob": round(prob, 4),
            "ev": round(ev, 2),
            "kelly": round(kelly, 4),
        })

    # EV降順ソート → 上位N件
    results.sort(key=lambda x: x["ev"], reverse=True)
    top_bets = results[:top_n]

    print(f"\n  🏆 AI推奨 TOP{top_n} 買い目:")
    print(f"  {'券種':>6} {'組合せ':>12} {'オッズ':>8} {'AI確率':>8} {'EV':>6} {'Kelly':>8}")
    print("  " + "-" * 55)
    for bet in top_bets:
        print(
            f"  {bet['type']:>6} {bet['targets']:>12} "
            f"{bet['odds']:>8.1f} {bet['prob']:>7.4f} "
            f"{bet['ev']:>6.2f} {bet['kelly']:>7.4f}"
        )

    # --- 出力 ---
    if output_path is None:
        base = Path(race_json_path)
        output_path = base.parent / f"{base.stem}_top_bets.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(top_bets, f, ensure_ascii=False, indent=2)

    print(f"\n  💾 {output_path} 保存完了（{len(top_bets)}件）")
    return top_bets


def main() -> None:
    """CLI エントリーポイント。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ExcelオッズからAI推奨TOP10買い目を生成"
    )
    parser.add_argument(
        "--excel", required=True, help="netkeibaオッズExcelのパス"
    )
    parser.add_argument(
        "--race", required=True, help="レースID（例: tulip2026）"
    )
    parser.add_argument(
        "--top", type=int, default=10, help="出力件数（デフォルト: 10）"
    )
    args = parser.parse_args()

    races_dir = BASE_DIR / "frontend" / "public" / "races"
    race_json_path = races_dir / f"{args.race}.json"

    if not race_json_path.exists():
        print(f"ERROR: {race_json_path} が見つかりません")
        print(f"  先に predict_race.py を実行してください")
        sys.exit(1)

    if not Path(args.excel).exists():
        print(f"ERROR: {args.excel} が見つかりません")
        sys.exit(1)

    print("=" * 60)
    print(f"SAKURA ORACLE — AI推奨買い目生成")
    print(f"  Excel: {args.excel}")
    print(f"  Race: {args.race}")
    print("=" * 60)

    output_path = races_dir / f"{args.race}_top_bets.json"
    generate_top_bets(args.excel, race_json_path, output_path, top_n=args.top)

    print("\n完了!")


if __name__ == "__main__":
    main()
