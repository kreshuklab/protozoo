import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from protozoo.config_base import BaseConfig, HIDE, Backend, Representation

BACKEND = Backend(os.environ.get("BACKEND", "pytorch"))

if BACKEND == Backend.PYTORCH:
    from protozoo.config_pytorch import ModelConfig, OptimizerConfig, LossConfig, Callback, LogConfig
else:
    raise NotImplementedError(BACKEND)


@dataclass
class ModelZooEntry(BaseConfig):
    """
    A model zoo entry is a collection of (nested) Configurations that are derived from BaseConfig.
    Each backend adapts the shared configs, which allow for a common string representation.
    """
    origin: str = Path(__file__).as_uri()
    repr=Representation(BACKEND)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    trainer_callbacks: List[Callback] = field(default_factory=list, metadata={HIDE: True})
    evaluator_callbacks: List[Callback] = field(default_factory=list, metadata={HIDE: True})
    predictor_callbacks: List[Callback] = field(default_factory=list, metadata={HIDE: True})
    log_config: LogConfig = field(default_factory=LogConfig)


if __name__ == "__main__":
    entry = ModelZooEntry(repr=Representation.STRING)

    print("\nfirst entry")
    print(entry)
    print("\nas dict")
    print(entry.as_dict())
    print()
    entry = entry.get_mapped(Representation.PYTORCH)
    print("\nafter mapping")
    print(entry)
    print("\nas dict")
    print(entry.as_dict())

    entry.save(Path.home() / "protozoo" / "string.yaml", repr=Representation.STRING)
    entry.save(Path.home() / "protozoo" / "pytorch.yaml", repr=Representation.PYTORCH, save_hidden=True, safe_dump=False)
