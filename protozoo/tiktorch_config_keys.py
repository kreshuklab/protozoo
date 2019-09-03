import torch.nn
import warnings

from dataclasses import dataclass, field
from typing import Type, Any, Optional, Callable, Mapping
from ignite.engine import Events, Engine


default_optimizer_class = torch.optim.Adam
default_loss_class = torch.nn.MSELoss


@dataclass
class ModelConfig:
    model_class: Type
    pretrained_source: Optional[Any] = None
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.pretrained_source is not None:
            raise NotImplementedError("pretrained model source")


@dataclass
class OptimizerConfig:
    optimizer_class: Optional[Type] = None
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.optimizer_class is None:
            warnings.warn(f"Using default optimizer: {default_optimizer_class}")
            self.optimizer_class = default_optimizer_class


@dataclass
class LossConfig:
    loss_class: Optional[Type] = None
    loss_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.loss_class is None:
            warnings.warn(f"Using default loss class: {default_loss_class}")
            self.loss_class = default_loss_class


@dataclass
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
    trainer_callbacks: Callbacks = field(default_factory=Callbacks)
    evaluator_callbacks: Callbacks = field(default_factory=Callbacks)
    predictor_callbacks: Callbacks = field(default_factory=Callbacks)
