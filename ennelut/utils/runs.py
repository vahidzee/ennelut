from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import tqdm
import wandb
import yaml


# functions
def load_yaml(path: str):
    with open(path) as f:
        return yaml.load(f, yaml.FullLoader)


def save_yaml(path: str, data: dict):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def read_and_average_results(path: str):
    results = load_yaml(path)
    return {k: np.array(v).mean().item() for k, v in results.items()}


def load_sweep_runs(identifier: str, id_function: Callable[[wandb.wandb_sdk.wandb_run.Run], Any]):
    # read the runs in the correspnding sweep
    api = wandb.Api(timeout=19)
    sweep = api.sweep(identifier)

    runs = defaultdict(list)
    for run in sweep.runs:
        identifier = id_function(run)
        if identifier is None:
            continue
        runs[identifier].append(run)
    return runs


def download_artifacts(run: wandb.wandb_sdk.wandb_run.Run, slug: str, path: str = ""):
    for file in run.files():
        if (
            file.name.startswith("checkpoints/last")
            or file.name.startswith("evaluation")
            or file.name.startswith("config")
        ):
            file.download(os.path.join(path, slug), replace=True)


def aggregate_stop_results(results: dict, aggregate: dict | None = None):
    if aggregate is None:
        aggregate = dict()
    for k, v in results.items():
        if k not in aggregate:
            aggregate[k] = []
        aggregate[k].append(v)
    return aggregate


def crawl_scaling_results(root_dir, stops):
    results = dict()
    subdirs = sorted([x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))])
    for stop in tqdm.tqdm(stops):
        # find runs for the stop
        runs = []
        for subdir in subdirs:
            if subdir.split("-")[0] == stop:
                runs.append(subdir)
        aggregate = dict(hald=dict(), images=dict())
        for run in runs:
            run_results_dir = os.path.join(root_dir, run, "evaluation")
            run_results_files = [x for x in os.listdir(run_results_dir) if x.endswith(".yaml")]
            image_results = [x for x in run_results_files if x != "hald-256.yaml"]
            aggregate["hald"] = aggregate_stop_results(
                read_and_average_results(os.path.join(run_results_dir, "hald-256.yaml")), aggregate["hald"]
            )
            aggregate_images = dict()
            for file in image_results:
                aggregate_images = aggregate_stop_results(
                    read_and_average_results(os.path.join(run_results_dir, file)), aggregate_images
                )
            aggregate_images = {k: np.array(v) for k, v in aggregate_images.items()}
            aggregate_images = {k: np.where(np.isinf(v), 100, v).mean().item() for k, v in aggregate_images.items()}
            aggregate["images"] = aggregate_stop_results(aggregate_images, aggregate["images"])
        results[stop] = aggregate
    return results
