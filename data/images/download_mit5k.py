from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By

MIT5K_URL = "https://data.csail.mit.edu/graphics/fivek/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected MIT-Adobe FiveK DNG images by image ID.")
    parser.add_argument(
        "--ids",
        nargs="*",
        type=int,
        default=None,
        help="Image IDs to download. Accepts space-separated IDs.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="Text file containing image IDs separated by commas, whitespace, or newlines.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where downloaded images should be stored. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--page-url",
        default=MIT5K_URL,
        help="MIT-Adobe FiveK index URL.",
    )
    parser.add_argument("--headful", action="store_true", help="Run Chrome with a visible browser window.")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to wait between download clicks.")
    parser.add_argument(
        "--load-images", action="store_true", help="Allow browser image loading while crawling the table."
    )
    return parser.parse_args()


def load_ids(ids: list[int] | None = None, ids_file: Path | None = None) -> set[int]:
    selected_ids = set(ids or [])
    if ids_file is not None:
        text = ids_file.expanduser().read_text()
        selected_ids.update(int(value) for value in re.findall(r"\d+", text))
    if not selected_ids:
        raise ValueError("Provide at least one image ID via --ids or --ids-file.")
    return selected_ids


def build_browser(output_dir: Path, headless: bool = True, load_images: bool = False) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    prefs = {
        "download.default_directory": str(output_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    }
    if not load_images:
        prefs["profile.managed_default_content_settings.images"] = 2
    options.add_experimental_option("prefs", prefs)
    return webdriver.Chrome(options=options)


def download_images(
    browser: webdriver.Chrome,
    page_url: str,
    image_ids: set[int],
    delay: float = 0.2,
) -> set[int]:
    browser.get(page_url)
    table = browser.find_element(by=By.XPATH, value="/html/body/table")
    rows = table.find_elements(by=By.XPATH, value="./tbody/tr")
    downloaded_ids = set()

    for row in rows:
        cells = row.find_elements(by=By.XPATH, value="./td")
        if not cells:
            continue
        try:
            image_id = int(cells[0].text)
        except ValueError:
            continue
        if image_id not in image_ids:
            continue

        link = row.find_element(by=By.XPATH, value="./td[3]/a")
        link.click()
        downloaded_ids.add(image_id)
        print(f"downloaded {image_id}")
        if delay > 0:
            time.sleep(delay)

    return downloaded_ids


def main() -> None:
    args = parse_args()
    image_ids = load_ids(ids=args.ids, ids_file=args.ids_file)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    browser = build_browser(output_dir=output_dir, headless=not args.headful, load_images=args.load_images)
    try:
        downloaded_ids = download_images(
            browser=browser,
            page_url=args.page_url,
            image_ids=image_ids,
            delay=args.delay,
        )
    finally:
        browser.close()

    missing_ids = sorted(image_ids - downloaded_ids)
    if missing_ids:
        print(f"missing IDs: {missing_ids}")


if __name__ == "__main__":
    main()
