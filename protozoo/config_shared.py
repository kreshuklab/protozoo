import numpy

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any, Tuple, Type, Mapping, Union


from protozoo.config_base import BaseConfig, HIDE

DEFAULT = "default"


class MiniBatch:
    """
    Base class for different mini batch formats. Mainly used for type checking
    """

    pass


@dataclass
class NumpyMiniBatchOfInput(MiniBatch):
    """
    A minimalistic mini batch format consisting of a single numpy.ndarray input
    note: only suitable for prediction, not training or evaluation
    """

    input_array: numpy.ndarray


@dataclass
class NumpyMiniBatchOfInputTarget(NumpyMiniBatchOfInput):
    """
    The standard mini batch format with numpy.ndarray(s): input, target
    """

    target_array: numpy.ndarray


@dataclass
class SharedModelConfig(BaseConfig):
    model_class: Union[str, Type] = DEFAULT
    pretrained_source: Optional[Any] = field(default=None, metadata={HIDE: True})
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)

    create_model_input: Callable[[MiniBatch], Tuple[Any, ...]] = field(default=DEFAULT, metadata={HIDE: True})
    create_loss_input: Callable[[Any, MiniBatch], Tuple[Any, ...]] = field(default=DEFAULT, metadata={HIDE: True})


@dataclass
class SharedOptimizerConfig(BaseConfig):
    optimizer_class: Union[str, Type] = DEFAULT
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SharedLossConfig(BaseConfig):
    loss_class: Union[Type, str] = DEFAULT
    loss_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SharedLogConfig(BaseConfig):
    dir: Path = field(default=Path.home() / "protozoo", metadata={HIDE: True})
    model_save_interval: int = 2
    model_n_saved: int = 2


@dataclass
class SharedCallback(BaseConfig):
    event: Any = None
    function: Callable = lambda *args, **kwargs: None
