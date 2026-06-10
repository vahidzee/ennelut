from __future__ import annotations

import dypy
import matplotlib
import torch
from jsonargparse import ArgumentParser

import ennelut
from ennelut.training.config import Config
from ennelut.training.loop import run_experiment
from ennelut.training.sweep import Sweep

torch.set_float32_matmul_precision("medium")

dypy.register_context(ennelut)
matplotlib.use("agg")


if __name__ == "__main__":
    # create the argument parser
    parser = ArgumentParser()
    subparsers = parser.add_subcommands(dest="command", required=False)
    sweep_parser = subparsers.add_subcommand("sweep", parser=Sweep.populate_parser(ArgumentParser()))
    individual_parser = subparsers.add_subcommand("individual", parser=Config.populate_parser(ArgumentParser()))
    args = parser.parse_args()

    if args.command == "sweep":
        sweep = dypy.functions.call_with_dynamic_args(Sweep, **args.sweep)
        sweep_id = sweep.initialize()
        if not sweep.submit_only:
            sweep.run(run_experiment)
    else:
        run_experiment(config=args.individual.as_dict())
