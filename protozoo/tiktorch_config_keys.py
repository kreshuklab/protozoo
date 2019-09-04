import torch.nn

from dataclasses import dataclass, field
from typing import Type, Any, Optional, Callable, Mapping, Tuple, List
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from pathlib import Path

default_optimizer_class = torch.optim.Adam
default_loss_class = torch.nn.MSELoss


class MiniBatch:
    """
    Base class for different mini batch formats. Mainly used for type checking
    """

    pass


@dataclass
class TorchMiniBatchOfInputTensor(MiniBatch):
    """
    A minimalistic mini batch format consisting of a single torch.Tensor input
    note: only suitable for prediction, not training or evaluation
    """

    input_tensor: torch.Tensor


@dataclass
class TorchMiniBatchOfInputTensorTargetTensor(TorchMiniBatchOfInputTensor):
    """
    The standard mini batch format with torch.Tensor(s): input, target
    """

    target_tensor: torch.Tensor


def default_create_model_input(mini_batch: TorchMiniBatchOfInputTensor) -> Tuple[torch.Tensor]:
    """
    Extract the input torch.Tensor from a mini_batch of type `TorchMiniBatchOfInputTensor` (or inherited thereof)
    """

    return (mini_batch.input_tensor,)


def default_create_loss_input(
    model_output: torch.Tensor, mini_batch: TorchMiniBatchOfInputTensorTargetTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines the model output with the mini batch target.
    """

    return model_output, mini_batch.target_tensor


@dataclass
class ModelConfig:
    model_class: Type
    pretrained_source: Optional[Any] = None
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)

    create_model_input: Callable[[MiniBatch], Tuple[Any, ...]] = default_create_model_input
    create_loss_input: Callable[[Any, MiniBatch], Tuple[Any, ...]] = default_create_loss_input

    def __post_init__(self):
        if self.pretrained_source is not None:
            raise NotImplementedError("pretrained model source")


@dataclass
class OptimizerConfig:
    optimizer_class: Type = default_optimizer_class
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    loss_class: Type = default_loss_class
    loss_kwargs: Mapping[str, Any] = field(default_factory=dict)


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


@dataclass
class ModelZooEntry:
    model_config: ModelConfig
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    trainer_callbacks: List[Callback] = field(default_factory=list)
    evaluator_callbacks: List[Callback] = field(default_factory=list)
    predictor_callbacks: List[Callback] = field(default_factory=list)
