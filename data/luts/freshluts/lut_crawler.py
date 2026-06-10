from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

FRESHLUTS_URL = "https://freshluts.com/"
LOGIN_BUTTON_XPATH = "/html/body/nav/div/div/ul[2]/li[4]/a[2]"
DOWNLOAD_BUTTON_XPATH = "/html/body/div[1]/div[1]/div/div/div[2]/div[1]/a[1]"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FreshLUTs .cube files listed in a metadata CSV.")
    parser.add_argument("--username", required=True, help="FreshLUTs username or email.")
    parser.add_argument("--password", required=True, help="FreshLUTs password.")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path.home() / "Downloads",
        help="Directory where the browser downloads files. Defaults to ~/Downloads.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where validated .cube files are stored. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).with_name("metadata.csv"),
        help="CSV containing FreshLUTs metadata with ID and URL columns.",
    )
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep invalid downloaded .cube files in the download directory instead of deleting them.",
    )
    parser.add_argument("--headful", action="store_true", help="Run Chrome with a visible browser window.")
    parser.add_argument("--page-delay", type=float, default=1.5, help="Seconds to wait after opening each LUT page.")
    parser.add_argument("--download-delay", type=float, default=2.0, help="Seconds to wait after clicking download.")
    return parser.parse_args()


def build_browser(download_dir: Path, headless: bool = True) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
        },
    )
    return webdriver.Chrome(options=options)


def login(browser: webdriver.Chrome, username: str, password: str) -> None:
    browser.get(FRESHLUTS_URL)
    browser.find_element(by=By.XPATH, value=LOGIN_BUTTON_XPATH).click()
    time.sleep(1)

    browser.find_element(by=By.ID, value="user_email").send_keys(username)
    browser.find_element(by=By.ID, value="user_password").send_keys(password)
    browser.find_element(by=By.NAME, value="commit").click()
    time.sleep(1)


def find_downloaded_cube(download_dir: Path) -> Path | None:
    cube_files = sorted(download_dir.glob("*.cube"), key=lambda path: path.stat().st_mtime, reverse=True)
    return cube_files[0] if cube_files else None


def validate_cube(path: Path) -> bool:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from ennelut.utils import load_cube

    try:
        load_cube(str(path))
    except Exception:
        return False
    return True


def move_valid_cube(cube_path: Path, lut_id: int, target_dir: Path, keep_invalid: bool = False) -> bool:
    if not validate_cube(cube_path):
        if not keep_invalid:
            cube_path.unlink(missing_ok=True)
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(cube_path), target_dir / f"{lut_id}.cube")
    return True


def download_lut(
    browser: webdriver.Chrome,
    url: str,
    lut_id: int,
    download_dir: Path,
    target_dir: Path,
    page_delay: float,
    download_delay: float,
    keep_invalid: bool,
) -> bool:
    print(url, lut_id)
    browser.get(url)
    time.sleep(page_delay)
    browser.find_element(by=By.XPATH, value=DOWNLOAD_BUTTON_XPATH).click()
    time.sleep(download_delay)

    cube_path = find_downloaded_cube(download_dir)
    if cube_path is None:
        print(f"No .cube file downloaded for LUT {lut_id}.")
        return False
    return move_valid_cube(cube_path, lut_id=lut_id, target_dir=target_dir, keep_invalid=keep_invalid)


def update_metadata(metadata_path: Path, df: pd.DataFrame, downloaded_ids: set[int]) -> None:
    df = df[df["ID"].isin(downloaded_ids)]
    df.to_csv(metadata_path, index=False)


def main() -> None:
    args = parse_args()
    download_dir = args.download_dir.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()

    download_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path)
    browser = build_browser(download_dir=download_dir, headless=not args.headful)
    downloaded_ids: set[int] = set()

    try:
        login(browser, username=args.username, password=args.password)
        for _, row in df.iterrows():
            lut_id = int(row["ID"])
            if download_lut(
                browser=browser,
                url=row["URL"],
                lut_id=lut_id,
                download_dir=download_dir,
                target_dir=target_dir,
                page_delay=args.page_delay,
                download_delay=args.download_delay,
                keep_invalid=args.keep_invalid,
            ):
                downloaded_ids.add(lut_id)
    finally:
        browser.close()

    update_metadata(metadata_path=metadata_path, df=df, downloaded_ids=downloaded_ids)


if __name__ == "__main__":
    main()
