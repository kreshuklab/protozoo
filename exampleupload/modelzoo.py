from protozoo.config import ModelZooEntry, ModelConfig

from DUNet import DUNet2D


def get_entry():
    return ModelZooEntry(
        model_config=ModelConfig(model_class=DUNet2D, model_kwargs={"in_channels": 1, "out_channels": 1, "N": 2})
    )
