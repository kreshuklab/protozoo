import torch.nn
import warnings

from dataclasses import dataclass, field
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from pathlib import Path
from typing import Type, Any, Optional, Callable

default_optimizer_class = torch.optim.Adam
default_loss_class = torch.nn.MSELoss


class ModelConfig:
    def __init__(self, model_class: Type, pretrained_source: Optional[str] = None, **model_kwargs: Any):
        self.model_class = model_class
        self.pretrained_source = pretrained_source
        self.model_kwargs = model_kwargs


class OptimizerConfig:
    def __init__(self, optimizer_class: Optional[Type] = None, **optimizer_kwargs: Any):
        if optimizer_class is None:
            warnings.warn(f"Using default optimizer: {default_optimizer_class}")
            optimizer_class = default_optimizer_class

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs


class LossConfig:
    def __init__(self, loss_class: Optional[Type] = None, **loss_kwargs: Any):
        if loss_class is None:
            warnings.warn(f"Using default loss: {default_loss_class}")
            loss_class = default_loss_class

        self.loss_class = loss_class
        self.loss_kwargs = loss_kwargs


class LogConfig:
    def __init__(self, dir: Optional[Path] = None, save_interval: int = 2, n_saved: int = 2):
        if dir is None:
            dir = Path(Path.home() / "protozoo")
            warnings.warn(f"Logging to default directory: {dir}")

        self.dir = dir
        self.checkpointer: ModelCheckpoint = ModelCheckpoint(
            self.dir.as_posix(), "protozoo", save_interval=save_interval, n_saved=n_saved, create_dir=True
        )


class Callback:
    event: Events
    function: Callable[[Engine], None]


class Callbacks:
    def __init__(self, *callbacks: Callback):
        self.callbacks = callbacks

    def __iter__(self):
        return iter(self.callbacks)


@dataclass
class ModelZooEntry:
    model_config: ModelConfig
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    trainer_callbacks: Callbacks = field(default_factory=Callbacks)
    evaluator_callbacks: Callbacks = field(default_factory=Callbacks)
    predictor_callbacks: Callbacks = field(default_factory=Callbacks)
