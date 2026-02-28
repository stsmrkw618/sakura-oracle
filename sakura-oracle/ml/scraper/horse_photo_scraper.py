"""
SAKURA ORACLE — 馬写真ダウンローダー

netkeibaの馬ページから写真を1枚ダウンロードして保存する。

使い方:
    from ml.scraper.horse_photo_scraper import download_horse_photo
    path = download_horse_photo("2022105678", save_dir=Path("images"))
"""

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.scraper.race_scraper import safe_request


def download_horse_photo(horse_id: str, save_dir: Path) -> Path | None:
    """馬ページから写真を1枚ダウンロードして保存する。

    Args:
        horse_id: netkeibaの馬ID（9-10桁）
        save_dir: 保存先ディレクトリ

    Returns:
        保存したファイルのPath。失敗時はNone。
    """
    # 既にファイルがあればスキップ（キャッシュ）
    out_path = save_dir / f"{horse_id}.jpg"
    if out_path.exists():
        return out_path

    # 馬ページから写真URLを取得
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    content = safe_request(url)
    if content is None:
        return None

    soup = BeautifulSoup(content, "lxml")

    # 写真ボックス内の show_photo リンクを検索
    photo_url = None
    for img in soup.select(".db_photo_box img, .horse_photo img, .photo img"):
        src = img.get("src", "")
        if "show_photo" in src:
            # サムネイルをフルサイズに変換（tn=1 → 除去）
            photo_url = re.sub(r"[?&]tn=\d+", "", src)
            break

    # フォールバック: ページ内の全imgタグから show_photo を検索
    if photo_url is None:
        for img in soup.select("img"):
            src = img.get("src", "")
            if "show_photo" in src:
                photo_url = re.sub(r"[?&]tn=\d+", "", src)
                break

    if photo_url is None:
        return None

    # https化
    if photo_url.startswith("//"):
        photo_url = "https:" + photo_url
    elif not photo_url.startswith("http"):
        photo_url = "https://db.netkeiba.com" + photo_url

    # 画像バイナリをダウンロード
    img_data = safe_request(photo_url)
    if img_data is None or len(img_data) < 1000:
        # 1KB未満はエラーページの可能性
        return None

    # 保存
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(img_data)

    return out_path


if __name__ == "__main__":
    # テスト用: 引数にhorse_idを指定
    if len(sys.argv) < 2:
        print("使い方: py ml/scraper/horse_photo_scraper.py <horse_id> [save_dir]")
        print("例: py ml/scraper/horse_photo_scraper.py 2022105678 ./images")
        sys.exit(1)

    hid = sys.argv[1]
    sdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./images")
    result = download_horse_photo(hid, save_dir=sdir)
    if result:
        print(f"保存完了: {result}")
    else:
        print("写真取得失敗")
