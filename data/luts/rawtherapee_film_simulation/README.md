# RawTherapee Film Simulation

## Source

RawTherapee documents a Film Simulation collection on RawPedia:
https://rawpedia.rawtherapee.com/Film_Simulation

The collection we inspected was the RawTherapee Film Simulation Collection version `2015-09-20`, licensed under CC BY-SA 4.0.

## How To Acquire

Follow the RawPedia Film Simulation page to download the Hald CLUT archive. The files are Hald CLUT images, mostly PNG files in sRGB, rather than `.cube` LUTs. To use them with the current EnneLUT data path, convert the Hald CLUT images into `.cube` files or add a loader that can train from Hald images directly.

## What We Looked Into

This source contains 500+ additional film-simulation profiles and could be a good candidate for future experiments. We did not use it for any reported paper results, but it is worth investigating as a broader Hald CLUT benchmark or as an additional qualitative evaluation set.

## Notes

- The original archive describes the profiles as film simulations, including pushed, pulled, and faded variants.
- Trademarked film stock names in filenames are informational and come from the upstream collection.
- Because this collection is CC BY-SA 4.0, derivative datasets or converted files may carry share-alike obligations.
