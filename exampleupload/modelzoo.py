from pathlib import Path

from protozoo.config_base import Representation
from protozoo.entry import ModelZooEntry, ModelConfig

import DUNet


def get_entry():
    return ModelZooEntry(
        origin=Path(__file__).as_uri(),
        model_config=ModelConfig(
            custom_module=DUNet, model_class=DUNet.DUNet2D, model_kwargs={"in_channels": 1, "out_channels": 1, "N": 2}
        ),
    )


if __name__ == "__main__":
    from pathlib import Path

    entry = get_entry()
    print(entry)
    entry.save(Path.home() / "protozoo" / "exampleupload.yaml")
    print(ModelZooEntry.load(Path.home() / "protozoo" / "exampleupload.yaml", repr=Representation.PYTORCH))
