from __future__ import annotations

# This code is adapted from https://github.com/vahidzee/lightning-toolbox
# Copyright Vahid Zehtab (vahid@zehtab.me) 2022-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import dypy
import wandb
import yaml
from jsonargparse import ActionConfigFile, ArgumentParser

from .parameters import ParametersMapping


@dataclass
class Sweep:
    name: str | None = None
    entity: str | None = None
    project: str | None = None

    # reproducibility
    seed_everything: bool | int | None = field(default=True, repr=False)

    # sweep configurations
    # rerun_id: Optional[str] = field(default=None, repr=False)
    run_name: str | None = field(default=None, repr=False)  # TODO: use this to set the agent run name
    sweep_id: int | str | None = field(default=None, repr=False)  # TODO: use this to resume a sweep
    count: int | None = field(default=None, repr=False)

    # sweep objective
    method: str | None = field(default="grid", repr=False)
    metric: str | None = field(default=None, repr=False)
    goal: str | None = field(default="minimize", repr=False)

    parameters: dict | None = field(default=None, repr=False)

    sweep_config: str | dict | None = None  # a file or results of a separate sweep configuration
    base_config: str | dict | None = None  # a file or results of a separate base configuration

    submit_only: bool = field(default=False, repr=False)

    def __post_init__(self):
        # read out the sweep config if it is a separate file
        if self.sweep_config is not None and isinstance(self.sweep_config, str):
            self.sweep_config = self.load_config(self.sweep_config)

        # process the sweep config (map the terms to match the wandb sweep config)
        self.sweep_config = self.__process_sweep_config()

        # read out the base config if it is a separate file
        if self.base_config is not None and isinstance(self.base_config, str):
            self.base_config = self.load_config(self.base_config)

    def __process_sweep_config(self):
        sweep_config = self.sweep_config if self.sweep_config is not None else {}
        metric = sweep_config.get("metric", {})
        metric["name"] = metric.get("name", self.metric)
        metric["goal"] = metric.get("goal", self.goal)
        config = {
            "method": sweep_config.get("method", self.method),
            "metric": metric,
        }

        self.parameters_mapping: ParametersMapping = ParametersMapping(config.get("parameters", self.parameters))
        config["parameters"] = self.parameters_mapping.mapped

        for item in ["name", "project", "entity"]:
            if getattr(self, item) is not None:
                config[item] = getattr(self, item)
        return config

    def load_config(self, path: str):
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Config file {path} does not exist")
        if path.suffix == ".yaml" or path.suffix == ".yml":
            return yaml.load(path.read_text(), Loader=yaml.FullLoader)
        elif path.suffix == ".json":
            return json.loads(path.read_text())

    def translate_config(self, parameters: dict):
        if self.parameters_mapping is None:
            raise ValueError("Parameters mapping is not initialized")
        return self.parameters_mapping.remap(parameters, base=self.base_config)

    def initialize(self):
        if self.sweep_id is None:
            sweep_id = wandb.sweep(sweep=self.sweep_config, entity=self.entity, project=self.project)
            self.sweep_id = sweep_id
        return self.sweep_id

    def run(self, function: Callable, count: int | None = None):
        return wandb.agent(
            self.sweep_id,
            function=partial(dypy.dynamic_args_wrapper(function), sweep=self),
            count=count if count is not None else self.count,
            project=self.project,
            entity=self.entity,
        )

    @staticmethod
    def populate_parser(parser: ArgumentParser):
        parser.add_class_arguments(Sweep, fail_untyped=False, sub_configs=True)
        parser.add_argument("--config", action=ActionConfigFile)
        return parser
