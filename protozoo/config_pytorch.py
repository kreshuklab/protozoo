import torch

from dataclasses import dataclass, field
from typing import Callable
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint


from protozoo.config_shared import (
    HIDE,
    MiniBatch,
    SharedCallback,
    SharedLogConfig,
    SharedLossConfig,
    SharedModelConfig,
    SharedOptimizerConfig,
)


@dataclass
class TorchMiniBatchOfInput(MiniBatch):
    """
    A minimalistic mini batch format consisting of a single torch.Tensor input
    note: only suitable for prediction, not training or evaluation
    """

    input_tensor: torch.Tensor


@dataclass
class TorchMiniBatchOfInputTarget(TorchMiniBatchOfInput):
    """
    The standard mini batch format with torch.Tensor(s): input, target
    """

    target_tensor: torch.Tensor


@dataclass
class Callback(SharedCallback):
    event: Events = field(default=Events.STARTED)
    function: Callable[[Engine], None] = field(default=lambda e: None)


@dataclass
class LogConfig(SharedLogConfig):
    checkpointer: ModelCheckpoint = field(init=False, metadata={HIDE: True})

    def __post_init__(self, *args):
        self.checkpointer = ModelCheckpoint(
            (self.dir / "checkpoints").as_posix(),
            "protozoo",
            save_interval=self.model_save_interval,
            n_saved=self.model_n_saved,
            create_dir=True,
        )
        super().__post_init__(*args)


@dataclass
class LossConfig(SharedLossConfig):
    pass


@dataclass
class ModelConfig(SharedModelConfig):
    pass


@dataclass
class OptimizerConfig(SharedOptimizerConfig):
    pass
