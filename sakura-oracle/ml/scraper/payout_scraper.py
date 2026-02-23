"""
SAKURA ORACLE — 配当（払戻金）パーサー

キャッシュ済みのnetkeibaページから馬連・ワイド・三連複の配当データを抽出。
キャッシュにないレースはnetkeibaから再取得する。

使い方:
    PYTHONIOENCODING=utf-8 py ml/scraper/payout_scraper.py

出力:
    data/raw/payouts.json
"""

import json
import pickle
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.scraper.config import CACHE_DIR, RAW_DIR
from ml.scraper.race_scraper import safe_request


def _cache_key_for_url(url: str) -> str:
    """URLからキャッシュファイル名を生成（race_scraper.pyと同じロジック）"""
    return re.sub(r'[^a-zA-Z0-9]', '_', url)[-100:]


def _load_cached_html(race_id: str) -> BeautifulSoup | None:
    """race_idに対応するキャッシュHTMLをロード（DB版優先、race版フォールバック）"""
    urls = [
        f"https://db.netkeiba.com/race/{race_id}/",
        f"https://race.netkeiba.com/race/result.html?race_id={race_id}",
    ]
    for url in urls:
        cache_file = CACHE_DIR / f"{_cache_key_for_url(url)}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                content = pickle.load(f)
            return BeautifulSoup(content, "lxml"), url
    return None, None


def _parse_payout_int(text: str) -> int:
    """配当テキスト（例: '1,000円'）から整数を抽出"""
    cleaned = re.sub(r'[^\d]', '', text)
    return int(cleaned) if cleaned else 0


def _parse_combo_nums(text: str) -> str:
    """組合せ番号テキスト（例: '4 - 18'）をハイフン区切りに正規化"""
    nums = re.findall(r'\d+', text)
    return '-'.join(nums)


def _parse_db_format(soup: BeautifulSoup) -> dict:
    """db.netkeiba.com形式: table.pay_table_01 + th.uren/wide/sanfuku"""
    result = {"quinella": [], "wide": [], "trio": []}

    tables = soup.select("table.pay_table_01")
    for table in tables:
        for row in table.select("tr"):
            th = row.select_one("th")
            if not th:
                continue
            th_classes = th.get("class", [])
            tds = row.select("td")
            if len(tds) < 2:
                continue

            combo_text = tds[0].get_text(separator="|", strip=True)
            payout_text = tds[1].get_text(separator="|", strip=True)

            combos = [c.strip() for c in combo_text.split("|") if c.strip()]
            payouts = [p.strip() for p in payout_text.split("|") if p.strip()]

            if "uren" in th_classes:
                # 馬連: 通常1組
                for c, p in zip(combos, payouts):
                    result["quinella"].append({
                        "combo": _parse_combo_nums(c),
                        "payout": _parse_payout_int(p),
                    })
            elif "wide" in th_classes:
                # ワイド: 3組
                for c, p in zip(combos, payouts):
                    result["wide"].append({
                        "combo": _parse_combo_nums(c),
                        "payout": _parse_payout_int(p),
                    })
            elif "sanfuku" in th_classes:
                # 三連複: 1組
                for c, p in zip(combos, payouts):
                    result["trio"].append({
                        "combo": _parse_combo_nums(c),
                        "payout": _parse_payout_int(p),
                    })

    return result


def _parse_race_format(soup: BeautifulSoup) -> dict:
    """race.netkeiba.com形式: table.Payout_Detail_Table + tr.Umaren/Wide/Fuku3"""
    result = {"quinella": [], "wide": [], "trio": []}

    tables = soup.select("table.Payout_Detail_Table")
    for table in tables:
        for row in table.select("tr"):
            row_classes = row.get("class", [])
            payout_td = row.select_one("td.Payout")
            result_td = row.select_one("td.Result")

            if not payout_td or not result_td:
                continue

            # 配当テキスト（br区切り → 複数）
            payout_spans = payout_td.select("span")
            payout_texts = []
            for span in payout_spans:
                for part in span.get_text(separator="\n").split("\n"):
                    part = part.strip()
                    if part:
                        payout_texts.append(part)

            # 組合せ番号: <ul>内の<li><span>数字</span></li> を1組として取得
            uls = result_td.select("ul")

            if "Umaren" in row_classes:
                # 馬連: 1組のul
                if uls:
                    nums = [s.get_text(strip=True) for s in uls[0].select("span") if s.get_text(strip=True)]
                    combo = '-'.join(nums)
                    payout = _parse_payout_int(payout_texts[0]) if payout_texts else 0
                    if combo and payout:
                        result["quinella"].append({"combo": combo, "payout": payout})

            elif "Wide" in row_classes:
                # ワイド: 3組のul
                for i, ul in enumerate(uls):
                    nums = [s.get_text(strip=True) for s in ul.select("span") if s.get_text(strip=True)]
                    combo = '-'.join(nums)
                    payout = _parse_payout_int(payout_texts[i]) if i < len(payout_texts) else 0
                    if combo and payout:
                        result["wide"].append({"combo": combo, "payout": payout})

            elif "Fuku3" in row_classes:
                # 三連複: 1組のul
                if uls:
                    nums = [s.get_text(strip=True) for s in uls[0].select("span") if s.get_text(strip=True)]
                    combo = '-'.join(nums)
                    payout = _parse_payout_int(payout_texts[0]) if payout_texts else 0
                    if combo and payout:
                        result["trio"].append({"combo": combo, "payout": payout})

    return result


def parse_payouts(race_id: str) -> dict | None:
    """race_idの配当データを取得（キャッシュ → ネットワーク）"""
    soup, source_url = _load_cached_html(race_id)

    if soup is None:
        # キャッシュなし → DB版を取得
        url = f"https://db.netkeiba.com/race/{race_id}/"
        content = safe_request(url)
        if content is None:
            return None
        soup = BeautifulSoup(content, "lxml")
        source_url = url

    # HTML形式を判定してパース
    if soup.select("table.pay_table_01"):
        return _parse_db_format(soup)
    elif soup.select("table.Payout_Detail_Table"):
        return _parse_race_format(soup)

    return None


def main() -> None:
    """全レースの配当データを取得してJSONに保存"""
    import pandas as pd

    print("=" * 50)
    print("SAKURA ORACLE — 配当データ取得")
    print("=" * 50)

    csv_path = RAW_DIR / "race_results.csv"
    if not csv_path.exists():
        print("race_results.csv が見つかりません")
        return

    df = pd.read_csv(csv_path)
    race_ids = [str(int(rid)) for rid in df["race_id"].unique()]
    print(f"対象: {len(race_ids)}レース\n")

    payouts = {}
    success = 0
    failed = []

    for rid in race_ids:
        result = parse_payouts(rid)
        if result and (result["quinella"] or result["wide"] or result["trio"]):
            payouts[rid] = result
            success += 1
            q = len(result["quinella"])
            w = len(result["wide"])
            t = len(result["trio"])
            print(f"  {rid}: 馬連{q} ワイド{w} 三連複{t}")
        else:
            failed.append(rid)
            print(f"  {rid}: 配当データなし")

    # 保存
    out_path = RAW_DIR / "payouts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payouts, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 50}")
    print(f"成功: {success}/{len(race_ids)}レース")
    if failed:
        print(f"失敗: {len(failed)}件 — {failed}")
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
