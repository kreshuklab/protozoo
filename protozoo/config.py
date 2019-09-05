from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from protozoo.shared_config import MapTo, BaseConfig, HIDE

framework = "pytorch"

if framework == "pytorch":
    from protozoo.pytorch_config import ModelConfig, OptimizerConfig, LossConfig, Callback, LogConfig

    map_to = MapTo.PYTORCH
else:
    raise NotImplementedError(framework)


@dataclass
class ModelZooEntry(BaseConfig):
    model_config: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    trainer_callbacks: List[Callback] = field(default_factory=list, metadata={HIDE: True})
    evaluator_callbacks: List[Callback] = field(default_factory=list, metadata={HIDE: True})
    predictor_callbacks: List[Callback] = field(default_factory=list, metadata={HIDE: True})
    log_config: LogConfig = field(default_factory=LogConfig)


if __name__ == "__main__":
    entry = ModelZooEntry()

    print("\nfirst entry")
    print(entry)
    print("\nas dict")
    print(entry.as_dict())
    print()
    entry = entry.get_mapped(map_to=MapTo.PYTORCH)
    print("\nafter mapping")
    print(entry)
    print("\nas dict")
    print(entry.as_dict())

    entry.export(Path.home() / "protozoo" / "string.yaml", map_to=MapTo.STRING)
    entry.export(Path.home() / "protozoo" / "pytorch.yaml", map_to=MapTo.PYTORCH, export_hidden=True, safe_dump=False)
