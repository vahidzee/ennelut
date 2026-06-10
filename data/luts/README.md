# LUT Collections

During this project, we gathered and inspected LUTs from several freely available sources so we could quickly test the method on a broad range of color transformations. These sources are useful starting points if you want to acquire LUTs, convert them into a common format, and try EnneLUT on your own collection.

For the paper results, however, we used a stricter subset: all reported numbers and experiments were limited to Creative Commons licensed LUTs gathered from FreshLUTs. The other folders document additional sources we looked into while building the project, but they were not used for the quantitative experiments in the paper unless explicitly noted.

## Collections

- [FreshLUTs](./freshluts/README.md): Creative Commons LUTs gathered from FreshLUTs. This is the source used for the paper experiments.
- [Log Footage](./log_footage/README.md): free LUTs from Shutterstock's log-footage LUT collection. We inspected this source, but did not use it for reported paper numbers.
- [Juan Melara](./melara/README.md): print-film-emulation LUTs from Juan Melara. We inspected this source, but did not use it for reported paper numbers.
- [Purple11](./purple11/README.md): free creative LUTs attributed to Purple11. We inspected this source, but did not use it for reported paper numbers.
- [Wanderlust](./wanderlust/README.md): free log-footage LUTs from PremiumBeat. We inspected this source, but did not use it for reported paper numbers.
- [RawTherapee Film Simulation](./rawtherapee_film_simulation/README.md): a CC BY-SA Hald CLUT collection with 500+ additional film-simulation profiles. We did not run experiments on it, but it is a promising source for future evaluation.

## Tooling

- [`sort.sh`](./sort.sh): renames `.cube` files in a directory with a numeric prefix based on sorted filename order. This is useful when a downloaded pack has inconsistent naming and you want stable local ordering.
- [`freshluts/metadata_cralwer.py`](./freshluts/metadata_cralwer.py): crawls FreshLUTs metadata into a CSV file with LUT IDs, source URLs, tags, authors, and before/after image links.
- [`freshluts/lut_crawler.py`](./freshluts/lut_crawler.py): logs into FreshLUTs, downloads `.cube` files listed in the metadata CSV, validates them with the project LUT loader, and writes them to a target directory.
- [`freshluts/eda.html`](./freshluts/eda.html): static exploratory report for browsing FreshLUTs metadata and preview imagery.

## Reproducing Paper LUT Data

To reproduce the paper experiments, follow the [FreshLUTs collection notes](./freshluts/README.md) and use the FreshLUTs tooling to rebuild the metadata CSV and download the `.cube` files. The experiment configs expect a local LUT directory with stable paths, so after downloading or manually gathering files you may need to use [`sort.sh`](./sort.sh) or another deterministic renaming step to keep local file ordering consistent.

The other collections documented here are useful for trying the method on additional LUTs, but they are not part of the reported paper numbers.

## Notes

- Check the license of any LUT collection before using it in experiments or redistributing processed data.
- Some collections provide Hald CLUT images rather than `.cube` files. Those need conversion before they can be used by the current LUT loading path.
- The helper scripts in this directory are utilities for organizing or acquiring LUTs; they are not required to use the trained models.
