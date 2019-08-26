from protozoo.tiktorch_config_keys import ModelZooEntry, ModelConfig

from DUNet import DUNet2D


def get_entry():
    return ModelZooEntry(model_config=ModelConfig(model_class=DUNet2D, in_channels=1, out_channels=2))
