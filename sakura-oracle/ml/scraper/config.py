"""SAKURA ORACLE — スクレイパー設定（拡張版: 3歳牝馬重賞全般）"""

import pathlib

# === ディレクトリ設定 ===
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TEMPLATE_DIR = DATA_DIR / "template"
CACHE_DIR = pathlib.Path(__file__).resolve().parent / "cache"

for d in [RAW_DIR, TEMPLATE_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === User-Agent（自分のブラウザのものに置き換え推奨） ===
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}

# === 取得対象レース ===
# 「開催日 + レース名キーワード」でrace_idを特定する方式
# date: 開催日 (YYYYMMDD), keyword: レース名に含まれる文字列
TARGET_RACES: list[dict[str, str]] = [
    # --- 2026年 ---
    {"date": "20260412", "keyword": "桜花賞", "label": "桜花賞2026"},
    {"date": "20260301", "keyword": "チューリップ", "label": "チューリップ賞2026"},
    {"date": "20260308", "keyword": "フィリーズ", "label": "フィリーズレビュー2026"},
    {"date": "20260214", "keyword": "クイーン", "label": "クイーンC2026"},
    {"date": "20260111", "keyword": "フェアリー", "label": "フェアリーS2026"},
    {"date": "20251214", "keyword": "阪神ジュベナイル", "label": "阪神JF2025(2026世代)"},
    {"date": "20251025", "keyword": "アルテミス", "label": "アルテミスS2025(2026世代)"},
    {"date": "20251101", "keyword": "ファンタジー", "label": "ファンタジーS2025(2026世代)"},
    # --- 2025年 ---
    {"date": "20250413", "keyword": "桜花賞", "label": "桜花賞2025"},
    {"date": "20250308", "keyword": "チューリップ", "label": "チューリップ賞2025"},
    {"date": "20250309", "keyword": "フィリーズ", "label": "フィリーズレビュー2025"},
    {"date": "20250215", "keyword": "クイーンC", "label": "クイーンC2025"},
    {"date": "20250111", "keyword": "フェアリー", "label": "フェアリーS2025"},
    {"date": "20241208", "keyword": "阪神ジュベナイル", "label": "阪神JF2024(2025世代)"},
    {"date": "20241026", "keyword": "アルテミス", "label": "アルテミスS2024(2025世代)"},
    {"date": "20241102", "keyword": "ファンタジー", "label": "ファンタジーS2024(2025世代)"},
    {"date": "20250525", "keyword": "オークス", "label": "オークス2025"},
    {"date": "20250914", "keyword": "ローズ", "label": "ローズS2025"},
    {"date": "20250913", "keyword": "紫苑", "label": "紫苑S2025"},
    {"date": "20251019", "keyword": "秋華賞", "label": "秋華賞2025"},
    # --- 2024年 ---
    {"date": "20240407", "keyword": "桜花賞", "label": "桜花賞2024"},
    {"date": "20240309", "keyword": "チューリップ", "label": "チューリップ賞2024"},
    {"date": "20240310", "keyword": "フィリーズ", "label": "フィリーズレビュー2024"},
    {"date": "20240210", "keyword": "クイーンC", "label": "クイーンC2024"},
    {"date": "20240108", "keyword": "フェアリー", "label": "フェアリーS2024"},
    {"date": "20231210", "keyword": "阪神ジュベナイル", "label": "阪神JF2023(2024世代)"},
    {"date": "20231028", "keyword": "アルテミス", "label": "アルテミスS2023(2024世代)"},
    {"date": "20231104", "keyword": "ファンタジー", "label": "ファンタジーS2023(2024世代)"},
    {"date": "20240519", "keyword": "オークス", "label": "オークス2024"},
    {"date": "20240914", "keyword": "ローズ", "label": "ローズS2024"},
    {"date": "20240907", "keyword": "紫苑", "label": "紫苑S2024"},
    {"date": "20241013", "keyword": "秋華賞", "label": "秋華賞2024"},
    # --- 2023年 ---
    {"date": "20230409", "keyword": "桜花賞", "label": "桜花賞2023"},
    {"date": "20230304", "keyword": "チューリップ", "label": "チューリップ賞2023"},
    {"date": "20230312", "keyword": "フィリーズ", "label": "フィリーズレビュー2023"},
    {"date": "20230211", "keyword": "クイーンC", "label": "クイーンC2023"},
    {"date": "20230109", "keyword": "フェアリー", "label": "フェアリーS2023"},
    {"date": "20221211", "keyword": "阪神ジュベナイル", "label": "阪神JF2022(2023世代)"},
    {"date": "20221029", "keyword": "アルテミス", "label": "アルテミスS2022(2023世代)"},
    {"date": "20221105", "keyword": "ファンタジー", "label": "ファンタジーS2022(2023世代)"},
    {"date": "20230521", "keyword": "オークス", "label": "オークス2023"},
    {"date": "20230917", "keyword": "ローズ", "label": "ローズS2023"},
    {"date": "20230909", "keyword": "紫苑", "label": "紫苑S2023"},
    {"date": "20231015", "keyword": "秋華賞", "label": "秋華賞2023"},
    # --- 2022年 ---
    {"date": "20220410", "keyword": "桜花賞", "label": "桜花賞2022"},
    {"date": "20220305", "keyword": "チューリップ", "label": "チューリップ賞2022"},
    {"date": "20220313", "keyword": "フィリーズ", "label": "フィリーズレビュー2022"},
    {"date": "20220212", "keyword": "クイーンC", "label": "クイーンC2022"},
    {"date": "20220110", "keyword": "フェアリー", "label": "フェアリーS2022"},
    {"date": "20211212", "keyword": "阪神ジュベナイル", "label": "阪神JF2021(2022世代)"},
    {"date": "20211030", "keyword": "アルテミス", "label": "アルテミスS2021(2022世代)"},
    {"date": "20211106", "keyword": "ファンタジー", "label": "ファンタジーS2021(2022世代)"},
    {"date": "20220522", "keyword": "オークス", "label": "オークス2022"},
    {"date": "20220918", "keyword": "ローズ", "label": "ローズS2022"},
    {"date": "20220910", "keyword": "紫苑", "label": "紫苑S2022"},
    {"date": "20221016", "keyword": "秋華賞", "label": "秋華賞2022"},
    # --- 2021年 ---
    {"date": "20210411", "keyword": "桜花賞", "label": "桜花賞2021"},
    {"date": "20210306", "keyword": "チューリップ", "label": "チューリップ賞2021"},
    {"date": "20210314", "keyword": "フィリーズ", "label": "フィリーズレビュー2021"},
    {"date": "20210213", "keyword": "クイーンC", "label": "クイーンC2021"},
    {"date": "20210111", "keyword": "フェアリー", "label": "フェアリーS2021"},
    {"date": "20201213", "keyword": "阪神ジュベナイル", "label": "阪神JF2020(2021世代)"},
    {"date": "20201031", "keyword": "アルテミス", "label": "アルテミスS2020(2021世代)"},
    {"date": "20201107", "keyword": "ファンタジー", "label": "ファンタジーS2020(2021世代)"},
    {"date": "20210523", "keyword": "オークス", "label": "オークス2021"},
    {"date": "20210919", "keyword": "ローズ", "label": "ローズS2021"},
    {"date": "20210911", "keyword": "紫苑", "label": "紫苑S2021"},
    {"date": "20211017", "keyword": "秋華賞", "label": "秋華賞2021"},
    # --- 2020年 ---（2歳戦は2021世代として既登録のため春〜秋9種のみ）
    {"date": "20200412", "keyword": "桜花賞", "label": "桜花賞2020"},
    {"date": "20200307", "keyword": "チューリップ", "label": "チューリップ賞2020"},
    {"date": "20200315", "keyword": "フィリーズ", "label": "フィリーズレビュー2020"},
    {"date": "20200215", "keyword": "クイーンC", "label": "クイーンC2020"},
    {"date": "20200113", "keyword": "フェアリー", "label": "フェアリーS2020"},
    {"date": "20200524", "keyword": "オークス", "label": "オークス2020"},
    {"date": "20200920", "keyword": "ローズ", "label": "ローズS2020"},
    {"date": "20200912", "keyword": "紫苑", "label": "紫苑S2020"},
    {"date": "20201018", "keyword": "秋華賞", "label": "秋華賞2020"},
    # --- 2019年 ---
    {"date": "20190407", "keyword": "桜花賞", "label": "桜花賞2019"},
    {"date": "20190302", "keyword": "チューリップ", "label": "チューリップ賞2019"},
    {"date": "20190310", "keyword": "フィリーズ", "label": "フィリーズレビュー2019"},
    {"date": "20190211", "keyword": "クイーンC", "label": "クイーンC2019"},
    {"date": "20190112", "keyword": "フェアリー", "label": "フェアリーS2019"},
    {"date": "20181209", "keyword": "阪神ジュベナイル", "label": "阪神JF2018(2019世代)"},
    {"date": "20181027", "keyword": "アルテミス", "label": "アルテミスS2018(2019世代)"},
    {"date": "20181103", "keyword": "ファンタジー", "label": "ファンタジーS2018(2019世代)"},
    {"date": "20190519", "keyword": "オークス", "label": "オークス2019"},
    {"date": "20190915", "keyword": "ローズ", "label": "ローズS2019"},
    {"date": "20190907", "keyword": "紫苑", "label": "紫苑S2019"},
    {"date": "20191013", "keyword": "秋華賞", "label": "秋華賞2019"},
    # --- 2018年 ---
    {"date": "20180408", "keyword": "桜花賞", "label": "桜花賞2018"},
    {"date": "20180303", "keyword": "チューリップ", "label": "チューリップ賞2018"},
    {"date": "20180311", "keyword": "フィリーズ", "label": "フィリーズレビュー2018"},
    {"date": "20180212", "keyword": "クイーンC", "label": "クイーンC2018"},
    {"date": "20180107", "keyword": "フェアリー", "label": "フェアリーS2018"},
    {"date": "20171210", "keyword": "阪神ジュベナイル", "label": "阪神JF2017(2018世代)"},
    {"date": "20171028", "keyword": "アルテミス", "label": "アルテミスS2017(2018世代)"},
    {"date": "20171103", "keyword": "ファンタジー", "label": "ファンタジーS2017(2018世代)"},
    {"date": "20180520", "keyword": "オークス", "label": "オークス2018"},
    {"date": "20180916", "keyword": "ローズ", "label": "ローズS2018"},
    {"date": "20180908", "keyword": "紫苑", "label": "紫苑S2018"},
    {"date": "20181014", "keyword": "秋華賞", "label": "秋華賞2018"},
]

# === アクセス設定 ===
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
MIN_WAIT = 3.0
MAX_WAIT = 7.0
BACKOFF_BASE = 60
