# MIT-Adobe FiveK Images

All the images used throughout the paper have been downloaded directly from the [MIT5K reference website](https://data.csail.mit.edu/graphics/fivek/) (as `.dng` files). To reduce computational costs we use Adobe Photoshop (`v25.3`) to resize all images to have a longer side of 1024 pixels and save them as `sRGB` `.png` files.

The following is a list of the images used for each part of the paper:
- `Split 1`: holds a hundred randomly selected images which was used to evaluate the models throughout the paper when talking about evaluation on natural images. These images have the following ids:
    ```
    [44,62,169,384,460,464,476,543,563,588,717,751,793,801,950,1090,1155,1203,1210,1297,1312,1315,1328,1399,1439,1447,1472,1578,1581,1655,1659,1660,1661,1692,1762,1778,1782,1786,1799,1819,1832,1856,1875,1918,1939,1948,2109,2120,2150,2165,2182,2234,2269,2298,2526,2528,2536,2563,2609,2635,2662,2693,2701,2709,2749,2783,2801,2939,3070,3123,3166,3196,3321,3391,3462,3741,3756,3877,3900,3996,4076,4121,4164,4237,4364,4371,4373,4382,4431,4484,4502,4508,4599,4617,4627,4800,4803,4888,4889,4999]
    ```

- `Split 2`: holds a hundred ranomly selecetd images (not overlapping with split-1) which was used to build an empirical distribution of colors. These images have the following ids:
    ```
    [4,120,172,204,260,375,385,443,474,520,524,542,552,566,580,584,617,701,712,759,767,772,839,895,898,912,916,1030,1051,1083,1133,1170,1252,1323,1352,1362,1401,1463,1558,1575,1628,1629,1716,1728,1735,1741,1763,1876,1885,1949,1973,2005,2036,2152,2171,2253,2276,2277,2278,2370,2404,2444,2577,2655,2940,2962,2965,2973,3059,3108,3126,3152,3268,3271,3469,3509,3561,3630,3714,3786,3834,3872,3893,3976,3978,3981,4089,4100,4174,4211,4392,4415,4428,4467,4503,4780,4819,4990]
    ```

<!-- - Teaser figure: has three pictures with IDs `[0096, 1133, 2269]`.
- Figure 6: has ten pictures with IDs `[1661, ]`.
- Figure 7: has ten pictures with IDs `[]`.
- Figure 8: has one picture with ID `a`.
- Figure 9: has one picture with ID `a`.
- Figure 10: has 11 pictures with IDs `[]`.
- Figure 11: has 11 pictures with IDs `[]`. -->

The rest of the pictures are selected arbitrarily and are only used to qualitatively visualize the results throughout the paper.

## Tooling

Use [`download_mit5k.py`](./download_mit5k.py) to download selected MIT-Adobe FiveK `.dng` files by ID. The script uses Selenium, disables browser image loading by default, and writes downloads to the current working directory unless `--output-dir` is provided.

```bash
python download_mit5k.py --help
```

Examples:

```bash
python download_mit5k.py --ids 44 62 169 --output-dir ./raw
python download_mit5k.py --ids-file split_1_ids.txt --output-dir ./split-1/raw
```
