from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

FRESHLUTS_LUT_URL = "https://freshluts.com/luts/{lut_id}"
NOT_FOUND = "Not found"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl FreshLUTs metadata into a CSV file.")
    parser.add_argument("--start-id", type=int, default=1, help="First LUT ID to crawl.")
    parser.add_argument("--end-id", type=int, default=None, help="Last LUT ID to crawl. Defaults to crawling forever.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("metadata.csv"),
        help="Output CSV path. Defaults to metadata.csv next to this script.",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between requests.")
    parser.add_argument(
        "--stop-on-missing",
        action="store_true",
        help="Stop crawling when a LUT page cannot be fetched.",
    )
    parser.add_argument("--append", action="store_true", help="Append to an existing output CSV if it exists.")
    return parser.parse_args()


def get_lut_html(lut_id: int, timeout: float = 10.0) -> str | None:
    url = FRESHLUTS_LUT_URL.format(lut_id=lut_id)
    response = requests.get(url, timeout=timeout)
    if response.status_code == 200:
        return response.text
    return None


def text_or_default(node, default: str = NOT_FOUND) -> str:
    return node.get_text(strip=True) if node is not None else default


def absolute_freshluts_url(path: str) -> str:
    return path if path.startswith("http") else f"https://freshluts.com{path}"


def absolute_protocol_url(path: str) -> str:
    return path if path.startswith("http") else f"https:{path}"


def extract_lut_details_from_html(html_content: str, lut_id: int) -> dict[str, int | str]:
    soup = BeautifulSoup(html_content, "html.parser")

    likes_and_downloads_div = soup.find("div", class_="likesanddownloads")
    if likes_and_downloads_div:
        downloads, likes = map(int, likes_and_downloads_div.stripped_strings)
    else:
        downloads = likes = NOT_FOUND

    property_details = {
        name_div.get_text(strip=True): value_div.get_text(strip=True)
        for name_div, value_div in zip(
            soup.find_all("div", class_="propertyname"),
            soup.find_all("div", class_="propertyvalue"),
            strict=False,
        )
    }

    user_card_div = soup.find("div", class_="usercard")
    author_link = user_card_div.find_next("a")["href"] if user_card_div else None
    before_photo = soup.find("img", src=lambda value: value and "beforeimages" in value)
    after_photo = soup.find("img", src=lambda value: value and "afterimages" in value)

    return {
        "ID": lut_id,
        "URL": FRESHLUTS_LUT_URL.format(lut_id=lut_id),
        "LUT Name": text_or_default(soup.find("h2")),
        "Downloads": downloads,
        "Likes": likes,
        "Gamma": property_details.get("Gamma", NOT_FOUND),
        "Color": property_details.get("Color", NOT_FOUND),
        "Key": property_details.get("Key", NOT_FOUND),
        "Style": property_details.get("Style", NOT_FOUND),
        "Author Name": text_or_default(user_card_div)[3:] if user_card_div else NOT_FOUND,
        "Author Profile Link": absolute_freshluts_url(author_link) if author_link else NOT_FOUND,
        "Before Photo Link": absolute_protocol_url(before_photo["src"]) if before_photo else NOT_FOUND,
        "After Photo Link": absolute_protocol_url(after_photo["src"]) if after_photo else NOT_FOUND,
    }


def scrape_lut_details(
    start_id: int = 1,
    end_id: int | None = None,
    timeout: float = 10.0,
    delay: float = 0.0,
    stop_on_missing: bool = False,
) -> pd.DataFrame:
    lut_details = []
    lut_id = start_id

    while end_id is None or lut_id <= end_id:
        print("loading id", lut_id)
        html_content = get_lut_html(lut_id, timeout=timeout)
        if html_content:
            lut_details.append(extract_lut_details_from_html(html_content, lut_id))
        else:
            print("not found")
            if stop_on_missing:
                break

        if delay > 0:
            time.sleep(delay)
        lut_id += 1

    return pd.DataFrame(lut_details)


def write_metadata(df: pd.DataFrame, output_path: Path, append: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if append and output_path.exists():
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
        df = df.drop_duplicates(subset=["ID"], keep="last").sort_values("ID")
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    df = scrape_lut_details(
        start_id=args.start_id,
        end_id=args.end_id,
        timeout=args.timeout,
        delay=args.delay,
        stop_on_missing=args.stop_on_missing,
    )
    write_metadata(df, output_path=output_path, append=args.append)


if __name__ == "__main__":
    main()
