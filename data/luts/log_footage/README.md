# Log Footage LUTs

## Source

This collection points to the free log-footage LUTs published by Shutterstock:
https://www.shutterstock.com/blog/free-luts-for-log-footage

## How To Acquire

Download the LUT pack from the source page, extract the `.cube` files, and place them in this directory or another local LUT workspace. If you want deterministic local names, run the helper script from `data/luts`:

```bash
./sort.sh path/to/extracted/luts
```

## What We Looked Into

We inspected this collection as an additional source of freely available LUTs for development and exploratory testing. These LUTs were not used for the paper's reported quantitative experiments.

## Notes

- The first 13 LUTs on the source page were dropped during inspection because they contained negative or undefined values.
