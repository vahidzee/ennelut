# FreshLUTs

## Source

This collection comes from FreshLUTs:
https://freshluts.com/

## How To Acquire

Use `metadata_cralwer.py` to build a metadata CSV, then use `lut_crawler.py` to download and validate the `.cube` files. Both scripts expose command-line help:

```bash
python metadata_cralwer.py --help
python lut_crawler.py --help
```

The downloader requires a FreshLUTs account:

```bash
python lut_crawler.py --username "$FRESHLUTS_USERNAME" --password "$FRESHLUTS_PASSWORD" --target-dir ./luts
```

## What We Looked Into

FreshLUTs was the source used for the paper experiments. Although we explored several LUT sources during the project, all reported numbers and experiments were limited to Creative Commons licensed LUTs gathered from FreshLUTs.

## Tooling

- [`metadata_cralwer.py`](./metadata_cralwer.py): crawls LUT pages and writes `metadata.csv`.
- [`lut_crawler.py`](./lut_crawler.py): downloads and validates `.cube` files listed in the metadata CSV.
- [`eda.html`](./eda.html): a static exploratory data analysis page for inspecting the crawled FreshLUTs metadata, sorting/filtering entries, and reviewing before/after preview images. It is meant for dataset inspection and curation, not for training.

## Notes

- The metadata CSV is used to track LUT IDs and source URLs.
- Downloaded `.cube` files are validated with the project LUT loader before being moved into the target directory.
