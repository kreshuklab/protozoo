import torch.nn
import warnings

from dataclasses import dataclass, field
from typing import Type, Any, Optional, Callable, List
from ignite.engine import Events, Engine


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


@dataclass
class Callback:
    event: Events
    function: Callable[[Engine], None]


@dataclass
class ModelZooEntry:
    model_config: ModelConfig
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    trainer_callbacks: List[Callback] = field(default_factory=list)
    evaluator_callbacks: List[Callback] = field(default_factory=list)
    predictor_callbacks: List[Callback] = field(default_factory=list)
