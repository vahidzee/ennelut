from __future__ import annotations

# This code is abtained from https://github.com/vahidzee/lightning-toolbox
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
import copy
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import dypy

WANDB_SWEEP_KEYS = ("values", "value", "distribution", "probabilities", "value")
VALUES_ALIAS = "values_alias"
TARGET = "target"


def set_value(name: str, value: Any, context):
    """
    Sets a value for variable "name" in the context "context", or in the global context if "context" is None.

    Args:
        name (str): The name of the variable to set.
        value (typing.Any): The value to set.

    Returns:
        None
    """
    name = name or ""
    var, name = context, name
    for split in name.split(".")[:-1] if name else []:
        if isinstance(var, dict):
            var = var[split]
        elif isinstance(var, list):
            var = var[int(split)]
        else:
            var = getattr(var, split)
    if isinstance(var, dict):
        if not name.split(".")[-1] and isinstance(value, dict):
            # recursively merge the value dict with context
            for k, v in value.items():
                set_value(k, v, var)

        elif isinstance(var[name.split(".")[-1]], dict) and isinstance(value, dict):
            # merge the two dicts
            var[name.split(".")[-1]].update(value)
        else:
            var[name.split(".")[-1]] = value
    else:
        setattr(var, name.split(".")[-1], value)


@dataclass
class ParametersMapping:
    """
    ParametersMapping is a utility class that maps nested parameters definitions into a flattened dict with renamed identifiers,
    and potentially renamed values (aliases), and vice versa.

    This class is used internally by the Sweep class to map an easier to use parameters definition (similar to the one used in PyTorch Lightning)
    for defining wandb sweep parameters.

    For example, take the following parameters definition:

    ```yaml
    parameters:
        sub_key1.sub_key2:
            values: [1, 2, 3]
        sub_key1:
            sub_key3:
                identifier: identifier2
                values: [4, 5, 6]
                values_alias: [alias1, alias2, alias3]
    ```

    The above definition will be mapped to the following flattened dict (if shorten_identifiers is True):

    ```yaml
    parameters:
        sub_key2:
            values: [1, 2, 3]
        identifier2:
            values: [alias1, alias2, alias3]
    ```

    This flattened dict can be used to define a wandb sweep config, and the results of the sweep can be remapped back to the original
    structure (and potentially merged with a base config) using the `remap` method.

    Args:
        source: a nested dict containing the parameters definition
        leaf_keys: keys that define a leaf nodes in the source dict (default: WANDB_SWEEP_KEYS)
        sep: separator to use for flattening the source dict (or unflattening the identifiers in the source & target dicts) (default: ".")
        shorten_identifiers: if True, shorten the identifiers by using the last unique sub-keys (seperated with `sep`) as identifier (default: True)
            This is useful when the original structure is deep resulting in long and unreadable identifiers.
        shorten_identifiers_sep: separator to use for shortening the identifiers (default: "__")
            This is only used if shorten_identifiers is True, and when greedily shortening the identifiers,
            the sub-keys are joined using this separator to create a new unique identifier.

    Attributes:
        identifiers: holds the mapping from identifiers to flattened keys
        alias_mapping: holds the mapping from alias_values to original values (for each identifier)
        mapped: flattened and mapped source dict
    """

    source: dict  # source parameters to be mapped
    leaf_keys: Iterable[str] | None = WANDB_SWEEP_KEYS  # keys that define a leaf nodes in the source dict
    sep: str = "."  # separator to use for flattening the source dict (or unflattening the identifiers in the source & target dicts)
    shorten_identifiers: bool = (
        True  # if True, shorten the identifiers by using the last unique sub-keys (seperated with `sep`) as identifier
    )
    shorten_identifiers_sep: str = "__"  # separator to use for shortening the identifiers

    # internal variables
    identifiers: dict[str, str] = field(
        init=False, default_factory=dict
    )  # holds the mapping from identifiers to flattened keys
    alias_mapping: dict[str, dict[str, str]] = field(
        init=False, default_factory=partial(defaultdict, dict)
    )  # holds the mapping from alias_values to original values (for each identifier)
    mapped: dict = field(init=False, default_factory=dict)  # flattened and mapped source dict
    targets: dict = field(init=False, default_factory=dict)  # map dict value to target value

    def __post_init__(self):
        # convert leaf_keys to set for faster lookup
        self.leaf_keys = set(self.leaf_keys if self.leaf_keys is not None else [])
        self.__flatten_dict(self.source)
        self.__rename_identifiers()
        self.__rename_values_alias()

    def __flatten_dict(self, source_dict: dict, parent_key=""):
        for key, value in source_dict.items():
            new_key = f"{parent_key}{self.sep}{key}" if parent_key else key

            if isinstance(value, dict):
                # check if any of the keys in the value dict are in leaf_keys
                if len(self.leaf_keys.intersection(value.keys())) > 0:
                    target = value.pop("target", new_key)
                    self.mapped[new_key] = value
                    self.targets[new_key] = target
                else:
                    self.__flatten_dict(
                        source_dict=value,
                        parent_key=new_key,
                    )
            else:
                self.mapped[new_key] = value
                self.targets[new_key] = new_key

    def __rename_identifiers(self):
        results = dict()
        for key, value in self.mapped.items():
            if isinstance(value, dict) and "identifier" in value:
                identifier = value.pop("identifier")
                assert self.sep not in identifier, f"Identifier {identifier} cannot contain {self.sep} separator"
                if identifier in self.identifiers:
                    raise ValueError(
                        f"Identifier is used for {self.identifiers[identifier]} and cannot be reused for {key}"
                    )
                self.identifiers[identifier] = key  # store the identifier mapping for unmapping
                results[identifier] = value
            elif self.shorten_identifiers:
                sub_keys = key.split(self.sep)
                for i in range(1, len(sub_keys) + 1):
                    # keep the last i sub-keys as identifier
                    identifier = self.shorten_identifiers_sep.join(sub_keys[-i:])
                    if identifier in self.identifiers:
                        continue
                    assert self.sep not in identifier, f"Identifier {identifier} cannot contain {self.sep} separator"
                    self.identifiers[identifier] = key
                    results[identifier] = value
                    break
        self.mapped = results

    def __apply_alias(self, key, value: dict, alias: Callable | list | tuple):
        if "values" not in value:
            logging.warning(f"Values not found in {value}, skipping alias")
            return value
        if callable(alias):
            try:
                aliases = alias(values=value["values"])
            except Exception:
                aliases = [alias(value=value) for value in value["values"]]
        elif isinstance(alias, (list, tuple)):
            if len(alias) != len(value["values"]):
                logging.warning(f"Invalid number of aliases ({alias}), provided for {value['values']}, skipping alias")
                return value
            aliases = [alias[i] for i in range(len(value["values"]))]
        else:
            return value
        for idx, item in enumerate(aliases):
            if item in self.alias_mapping[key]:
                raise ValueError(f"Alias {item} is already used for {self.alias_mapping[key][item]}")
            self.alias_mapping[key][item] = value["values"][idx]
        value["values"] = aliases
        return value

    def __rename_values_alias(self):
        results = dict()
        for key, value in self.mapped.items():
            if not isinstance(value, dict):
                results[key] = value
                continue
            alias = dypy.eval(value.pop(VALUES_ALIAS, None))
            alias = dypy.dynamic_args_wrapper(alias) if callable(alias) else alias
            results[key] = self.__apply_alias(key, value, alias)

        self.mapped = results

    def remap(self, target: dict, base: dict | None = None):
        results = dict() if base is None else copy.deepcopy(base)
        for identifier, value in target.items():
            value = copy.deepcopy(value)
            # translate aliases to values if needed
            if identifier in self.alias_mapping and isinstance(value, dict) and "values" in value:
                value["values"] = [self.alias_mapping[identifier][v] for v in value["values"]]
            if identifier in self.alias_mapping:
                value = self.alias_mapping[identifier][value]

            if self.sep in identifier:
                # if key is a sub-key path, translate the first part to identifier if needed
                key = self.identifiers.get(identifier.split(self.sep)[0], identifier)
                key = self.targets.get(key, key)
                key = ".".join([key, ".".join(identifier.split(self.sep)[1:])])
            else:
                # translate identifier to key
                key = self.identifiers.get(identifier, identifier)
                key = self.targets.get(key, key)

            # unflatten the key
            sub_keys = key.split(self.sep)
            # create the sub-key branch in dict with empty scopes if needed
            if not sub_keys[0]:
                target_value = results
            else:
                target_value = results.get(sub_keys[0], dict())
                results[sub_keys[0]] = target_value
            for sub_key in sub_keys[1:]:
                if sub_key not in target_value:
                    target_value[sub_key] = dict()
                target_value = target_value[sub_key]
            set_value(key, value, results)

        return results
