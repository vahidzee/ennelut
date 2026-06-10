from __future__ import annotations

import glob
import os
import random
from collections.abc import Iterable

import torch

from ennelut.data.hald import generate_grid, generate_hald_image
from ennelut.data.trilinear import trilinear_interpolation
from ennelut.utils import load_cube, load_image, normalize


class LUT:
    def __init__(self, path: str, device: torch.device = "cpu"):
        assert path is not None and path.lower().endswith(".cube"), "path must be a .cube file"
        self.path = path
        self.grid_size, self.outputs = load_cube(path, device=device)
        self.device = device

    def to(self, device: str):
        self.outputs = self.outputs.to(device)
        self.device = device
        return self

    def __call__(self, inputs: torch.Tensor):
        original_shape = inputs.shape  # to restore later
        inputs = inputs.reshape(-1, 3)
        grid = generate_grid(grid_size=self.grid_size, meshgrid=False, device=inputs.device)
        results = trilinear_interpolation(inputs=inputs, grid=grid, outputs=self.outputs)
        return results.reshape(original_shape)

    # @cached(ignore=["inputs"], images=True)
    def transform_hald(self, grid_size: int | None = None, inputs: torch.Tensor | None = None):
        assert inputs is not None or grid_size is not None, "either inputs or grid_size must be provided"
        if inputs is None:
            inputs = torch.from_numpy(generate_hald_image(grid_size=grid_size)).float().to(self.device)
        return inputs, self(inputs)

    # @cached(ignore=["image"], images=True)
    def transform_image(self, path: str, image: torch.Tensor | None = None):
        if image is None:
            image = torch.from_numpy(load_image(path)).float().to(self.device)
        return self(image)

    def __repr__(self) -> str:
        results = f"path='{self.path}'"
        results += f", grid={self.grid_size}^3" if self.grid_size is not None else ""
        return f"{self.__class__.__name__}({results})"


class MultiLUTs:
    def __init__(
        self,
        path: str | list[str],
        batch_size: int = 4096,
        luts_per_batch: int | None = None,
        seed: int = 313,
        split: float | int | None = None,
        device: torch.device = "cpu",
        space: str | None = None,
    ):
        super().__init__()
        self.paths = self.process_paths(paths=path, split=split, seed=seed)
        self.luts = [LUT(path=_path, device=device) for _path in self.paths]

        self.batch_size = batch_size
        self.luts_per_batch = None
        if luts_per_batch is not None:
            self.luts_per_batch = luts_per_batch if len(self.luts) > luts_per_batch else None

        self.seed = seed
        self.device = device

        self.space: torch.Tensor = None
        self.weights: torch.Tensor = None
        if space is not None:
            self.space, self.weights = torch.load(space, map_location=device)
            print("Loaded color space with", len(self.space), "samples from", space)
        self.generator = None  # set in __iter__

    def to(self, device: torch.device):
        for lut in self.luts:
            lut.to(device)
        self.device = device
        if self.space is not None:
            self.space = self.space.to(device)
            self.weights = self.weights.to(device)
        return self

    @staticmethod
    def process_paths(paths: str | list[str], split: float | int | None = None, seed: int = 313):
        # run through the the path(s) and if any includes wildcards, replace them with the actual paths
        if isinstance(paths, str):
            paths = [paths]
        processed_paths = []
        for path in paths:
            if "*" in path:
                processed_paths.extend(glob.glob(path))
            else:
                processed_paths.append(path)

        # check if split is specified in data config
        if split is not None:
            print("Subsampling", split, "paths out of", len(processed_paths))
            # create a rng with the split seed
            rng = random.Random(seed)
            processed_paths = rng.sample(
                processed_paths,
                k=int(len(processed_paths) * split) if split < 1.0 else int(split),
            )

        # sort processed paths by name
        processed_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        return processed_paths

    def __call__(self, inputs: torch.Tensor, luts_idxs: Iterable[int] | int | None = None):
        luts_idxs = [luts_idxs] if isinstance(luts_idxs, int) else luts_idxs

        original_shape = inputs.shape
        inputs = inputs.reshape(-1, 3)
        luts = self.luts if luts_idxs is None else [self.luts[idx] for idx in luts_idxs]
        grid_sizes = torch.tensor([lut.grid_size for lut in luts])

        # batch over luts with the same grid_size
        results = []
        permutation = torch.empty(0)
        for grid_size in grid_sizes.unique():
            grid = generate_grid(grid_size=grid_size, meshgrid=False, device=inputs.device)  # memoized
            grid_idxs = torch.where(grid_sizes == grid_size)[0]

            lut_outputs = torch.stack([luts[idx].outputs for idx in grid_idxs])
            batch_results = trilinear_interpolation(inputs=inputs, grid=grid, outputs=lut_outputs)

            results.append(batch_results.unsqueeze(0) if batch_results.ndim == 2 else batch_results)
            permutation = torch.cat([permutation, grid_idxs])
        results = torch.cat(results)
        if len(luts) == 1:
            results.unsqueeze_(0)
        return results[torch.argsort(permutation)].reshape(len(luts), *original_shape)

    def next(self):
        # generate a random batch of samples
        if self.space is None:
            inputs = torch.randint(
                0, 255, size=(self.batch_size, 3), dtype=torch.float32, device=self.device, generator=self.generator
            )
            inputs.div_(255.0)
            inputs = normalize(inputs)
        else:
            # take random samples from space with weights
            idx = torch.multinomial(self.weights, self.batch_size, replacement=True, generator=self.generator)
            inputs = self.space[idx]

        # select the luts to use for this batch
        luts_idx = torch.randperm(len(self.luts), generator=self.generator, device=self.device)
        if self.luts_per_batch is not None:
            luts_idx = luts_idx[: self.luts_per_batch]

        # transform the inputs with the selected luts
        outputs = self(inputs, luts_idxs=luts_idx)

        # repeat the inputs and idx for each lut
        inputs = inputs.unsqueeze(0).repeat(len(luts_idx), 1, 1)
        luts_idx = luts_idx.repeat_interleave(self.batch_size, dim=0).reshape(
            len(luts_idx), self.batch_size
        )  # todo: drop this

        return inputs, outputs, luts_idx

    def __getitem__(self, idx):
        return self.luts[idx]

    def __len__(self):
        return len(self.luts)

    def reset_iterator(self):
        self.generator = torch.Generator(self.device).manual_seed(self.seed)
        self.position = 0

    def __repr__(self):
        luts = ", ".join([repr(lut) for lut in self.luts])
        luts = f"luts=[{luts}], " if luts else ""
        return f"IterableLuts({luts})"
