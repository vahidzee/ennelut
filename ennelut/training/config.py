from __future__ import annotations

import copy
from dataclasses import dataclass, field

import dypy
import lightning
import tqdm
from jsonargparse import ActionConfigFile, ArgumentParser
from lightning.pytorch.loggers import WandbLogger

import ennelut.utils.vectors as vectors
from ennelut.data import MultiLUTs
from ennelut.model import ExactResFlow
from ennelut.training.objective import l2_objective, l2_objective_lab
from ennelut.training.sweep import Sweep


class ArgSpace(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


@dataclass
class Data:
    dataset_args: dict = field(default_factory=dict)
    path: str | None = None
    batch_size: int | None = None
    luts_per_batch: int | None = None
    split: float | int | None = None
    split_seed: int = 313

    def __post_init__(self):
        # process paths, and splits here so that we can save the hyperparameters
        self.dataset_args = dict() if self.dataset_args is None else self.dataset_args
        self.dataset_args["path"] = self.dataset_args.get("path", self.path)
        self.dataset_args["path"] = MultiLUTs.process_paths(
            paths=self.dataset_args["path"], split=self.split, seed=self.split_seed
        )
        self.dataset_args["batch_size"] = self.dataset_args.get("batch_size", self.batch_size)
        self.dataset_args["luts_per_batch"] = self.dataset_args.get("luts_per_batch", self.luts_per_batch)

    @property
    def hyperparameters(self):
        return {"dataset_args": self.dataset_args}

    def setup(self, context):
        context.data = MultiLUTs(**self.dataset_args)


@dataclass
class TrainingConfig:
    # train loop configs
    max_steps: int
    progress_bar: bool = True

    lr: float = 1e-2

    optimizer: str = "torch.optim.Adam"
    embeddings_optimizer: str | None = None
    optimizer_args: dict = field(default_factory=dict)
    embeddings_optimizer_args: dict | None = None
    optimizers_frequency: int = 0

    lr_scheduler: str | None = None
    lr_scheduler_args: dict | None = field(default_factory=dict)
    embeddings_lr_scheduler: str | None = None
    embeddings_lr_scheduler_args: dict | None = None

    objective_args: dict = field(default_factory=dict)

    normalization_constant: float | None = None

    def __post_init__(self):
        # process the optimizer
        self.optimizer_args = dict() if self.optimizer_args is None else self.optimizer_args
        self.optimizer_args["lr"] = self.optimizer_args.get("lr", self.lr)
        self.embeddings_optimizer_args = (
            copy.deepcopy(self.optimizer_args)
            if self.embeddings_optimizer_args is None
            else self.embeddings_optimizer_args
        )
        self.embeddings_optimizer_args["lr"] = self.embeddings_optimizer_args.get("lr", self.lr)

        # process the lr scheduler args
        self.embeddings_lr_scheduler_args = (
            copy.deepcopy(self.lr_scheduler_args)
            if self.embeddings_lr_scheduler_args is None
            else self.embeddings_lr_scheduler_args
        )

        if self.normalization_constant is not None:
            vectors.LEVEL = self.normalization_constant
            print("Updated normalization constant to", vectors.LEVEL)

    @property
    def hyperparameters(self):
        hparams = {
            "max_steps": self.max_steps,
            "optimizer": self.optimizer,
            "optimizer_args": self.optimizer_args,
            "objective_args": self.objective_args,
        }
        if self.embeddings_optimizer is not None:
            hparams["embeddings_optimizer"] = self.embeddings_optimizer
            hparams["embeddings_optimizer_args"] = self.embeddings_optimizer_args
        if self.lr_scheduler is not None:
            hparams["lr_scheduler"] = self.lr_scheduler
            hparams["lr_scheduler_args"] = self.lr_scheduler_args
        if self.embeddings_lr_scheduler is not None:
            hparams["embeddings_lr_scheduler"] = self.embeddings_lr_scheduler
            hparams["embeddings_lr_scheduler_args"] = self.embeddings_lr_scheduler_args
        if self.normalization_constant is not None:
            hparams["normalization_constant"] = self.normalization_constant
        return hparams

    def setup_optimizers(self, context):
        model: ExactResFlow = context.model
        use_separate_optimizers = self.embeddings_optimizer is not None and model.conditions_embedding is not None
        # setup optimizers
        context.optimizer = dypy.eval(self.optimizer)(
            model.parameters() if use_separate_optimizers is None else model.transforms.parameters(),
            **self.optimizer_args,
        )
        if use_separate_optimizers:
            context.embeddings_optimizer = dypy.eval(self.embeddings_optimizer)(
                [model.conditions_embedding], **self.embeddings_optimizer_args
            )
        else:
            context.embeddings_optimizer = None
            self.embeddings_optimizer = None
            self.embeddings_lr_scheduler = None
            self.optimizers_frequency = 0  # disable optimizers frequency

        context.optimizers = (
            [context.optimizer, context.embeddings_optimizer]
            if context.embeddings_optimizer is not None
            else [context.optimizer]
        )
        # handle frequency
        context.optimizers_frequency = self.optimizers_frequency
        context.optimizers_is_active = [True for _ in context.optimizers]

        if context.optimizers_frequency > 0 and len(context.optimizers) > 1:
            context.optimizers_is_active = [False for _ in context.optimizers]
            context.optimizers_is_active[0] = True
        elif context.optimizers_frequency < 0 and len(context.optimizers) > 1:
            context.optimizers_is_active = [True for _ in context.optimizers]
            context.optimizers_is_active[0] = False
            context.optimizers_frequency = abs(context.optimizers_frequency)

        # setup lr schedulers
        context.lr_schedulers = [None, None]
        context.lr_scheduler, context.embeddings_lr_scheduler = None, None

        if self.lr_scheduler is not None:
            context.lr_scheduler = dypy.eval(self.lr_scheduler)(context.optimizer, **self.lr_scheduler_args)
            context.lr_schedulers[0] = context.lr_scheduler

        if context.embeddings_optimizer is not None and self.embeddings_lr_scheduler is not None:
            context.embeddings_lr_scheduler = dypy.eval(self.embeddings_lr_scheduler)(
                context.embeddings_optimizer, **self.embeddings_lr_scheduler_args
            )
            context.lr_schedulers[1] = context.embeddings_lr_scheduler

    def setup_progress(self, context):
        context.step = 0
        context.max_steps = self.max_steps
        if self.progress_bar:
            context.progress_bar = tqdm.tqdm(
                total=context.max_steps, desc="training", dynamic_ncols=True, leave=True, position=0
            )
        else:
            context.progress_bar = None

    def setup(self, context):
        self.setup_optimizers(context=context)
        if self.objective_args.get("lab", False):
            context.objective = l2_objective_lab
        else:
            context.objective = l2_objective  # LUFlowObjective(**self.objective_args)
        self.setup_progress(context=context)


@dataclass
class Config:
    model: dict
    data: Data
    training: TrainingConfig
    fabric: dict | None = field(default_factory=dict)
    seed_everything: int = 42
    logger: str = "lightning.pytorch.loggers.WandbLogger"
    logger_args: dict = field(default_factory=dict)
    stages = ["setup", "train", "test"]

    checkpoint: str | None = None

    def __post_init__(self):
        self.data = Data(**self.data) if isinstance(self.data, dict) else self.data
        self.training = TrainingConfig(**self.training) if isinstance(self.training, dict) else self.training
        self.model = {**self.model, "conditions": len(self.data.dataset_args["path"])}

    @property
    def hyperparameters(self):
        return {
            "model": self.model,
            "data": self.data.hyperparameters,
            "training": self.training.hyperparameters,
            "fabric": self.fabric,
            "seed_everything": self.seed_everything,
        }

    def setup_fabric(self, context, logger=None):
        # setup callbacks
        callbacks = []
        for callback in self.fabric.get("callbacks", []):
            assert isinstance(callback, dict), "callback must be a dict"
            assert "class_path" in callback, "callback must have a class key"
            callbacks.append(dypy.eval(callback["class_path"])(**callback.get("init_args", {})))

        # setup fabric
        context.fabric = lightning.Fabric(**{**self.fabric, "callbacks": callbacks}, loggers=logger)

    def initiate_fabric(self, context):
        context.model = context.fabric.setup_module(context.model)
        context.data = context.data.to(context.fabric.device)
        context.optimizers = context.fabric.setup_optimizers(*context.optimizers)

        if isinstance(context.optimizers, (tuple, list)) and len(context.optimizers) == 2:
            context.optimizer = context.optimizers[0]
            context.embeddings_optimizer = context.optimizers[1]
        else:
            context.optimizer = (
                context.optimizers[0] if isinstance(context.optimizers, (tuple, list)) else context.optimizers
            )
            context.optimizers = [context.optimizer]

    def setup_logger(self, context):
        context.logger = dypy.eval(self.logger)(**self.logger_args)
        return context.logger

    def setup(self, logger=None):
        # create an argspace for the training context
        context = ArgSpace()
        context.logger = self.setup_logger(context=context) if logger is None else logger

        lightning.seed_everything(self.seed_everything)

        self.data.setup(context=context)
        context.model = ExactResFlow(**self.model)
        self.training.setup(context=context)
        self.setup_fabric(context=context, logger=context.logger)
        context.config = self
        return context

    @staticmethod
    def initialize(sweep: Sweep | None, config: dict | None = None):
        assert sweep is not None or config is not None, "either sweep or config must be specified"
        logger = None
        if sweep is not None:
            logger = WandbLogger(project=sweep.project, entity=sweep.entity, name=sweep.run_name)
            config = sweep.translate_config(dict(logger.experiment.config))
            config = Config(**config)
        else:
            config = dypy.functions.call_with_dynamic_args(Config, **config)
        return config.setup(logger=logger)

    @staticmethod
    def populate_parser(parser: ArgumentParser):
        parser.add_class_arguments(Config, fail_untyped=False, sub_configs=True)
        parser.add_argument("--config", action=ActionConfigFile)
        return parser

    def load_checkpoint(self, context):
        state = {"model": context.model, "optimizer": context.optimizer}
        if context.lr_scheduler is not None:
            state["lr_scheduler"] = context.lr_scheduler
        if context.embeddings_optimizer is not None:
            state["embeddings_optimizer"] = context.embeddings_optimizer
        if context.embeddings_lr_scheduler is not None:
            state["embeddings_lr_scheduler"] = context.embeddings_lr_scheduler

        if self.checkpoint is not None:
            res = context.fabric.load(self.checkpoint, state=state)
            context.step = res["step"]
            if context.progress_bar is not None:
                context.progress_bar.update(context.step)
            print(f"Loaded checkpoint at step {context.step}")
