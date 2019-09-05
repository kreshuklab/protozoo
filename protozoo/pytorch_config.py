import torch.nn

from dataclasses import dataclass, field
from typing import Type, Any, Optional, Callable, Mapping, Tuple, Union
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from pathlib import Path

from protozoo.shared_config import BaseConfig, MiniBatch, HIDE, MapTo


# todo: move most of it to shared_config, if dynamic dependency on torch is feasible...


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
class ModelConfig(BaseConfig):
    model_class: Optional[Type] = field(default=None, metadata={})
    pretrained_source: Optional[Any] = field(default=None, metadata={HIDE: True})
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)

    create_model_input: Callable[[MiniBatch], Tuple[Any, ...]] = field(
        default=default_create_model_input,
        metadata={
            HIDE: True,
            MapTo.STRING: lambda obj: obj.__name__,
            MapTo.PYTORCH: lambda name: globals().get(name, name),
        },
    )
    create_loss_input: Callable[[Any, MiniBatch], Tuple[Any, ...]] = field(
        default=default_create_loss_input,
        metadata={
            HIDE: True,
            MapTo.STRING: lambda obj: obj.__name__,
            MapTo.PYTORCH: lambda name: globals().get(name, name),
        },
    )


@dataclass
class OptimizerConfig(BaseConfig):
    optimizer_class: Union[str, Type] = field(
        default=torch.optim.Adam,
        metadata={MapTo.PYTORCH: lambda name: getattr(torch.optim, name), MapTo.STRING: lambda obj: obj.__name__},
    )
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig(BaseConfig):
    loss_class: Type = field(
        default=torch.nn.MSELoss,
        metadata={MapTo.PYTORCH: lambda name: getattr(torch.nn, name), MapTo.STRING: lambda obj: obj.__name__},
    )
    loss_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LogConfig(BaseConfig):
    dir: Path = field(default=Path.home() / "protozoo", metadata={HIDE: True})
    model_save_interval: int = 2
    model_n_saved: int = 2
    checkpointer: ModelCheckpoint = field(init=False, metadata={HIDE: True})

    def __post_init__(self, map_to):
        self.checkpointer = ModelCheckpoint(
            (self.dir / "checkpoints").as_posix(),
            "protozoo",
            save_interval=self.model_save_interval,
            n_saved=self.model_n_saved,
            create_dir=True,
        )
        super().__post_init__(map_to=map_to)


@dataclass
class Callback(BaseConfig):
    event: Events = field(default=Events.STARTED)
    function: Callable[[Engine], None] = field(default=lambda e: None)
