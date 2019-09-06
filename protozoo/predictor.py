import torch

from ignite.engine import Events, Engine
from typing import Optional

from protozoo.config_base import Representation
from protozoo.entry import ModelZooEntry


class Predictor(Engine):
    def __init__(self, model_zoo_entry: ModelZooEntry, model: Optional[torch.nn.Module] = None):
        model_zoo_entry = model_zoo_entry.get_mapped(Representation.PYTORCH)
        if model_zoo_entry.model_config.pretrained_source is not None:
            raise NotImplementedError("model_zoo_entry.model_config.pretrained_source")

        self.model_config = model_zoo_entry.model_config

        super().__init__(self.predict_batch)

        if model is not None and not isinstance(model, self.model_config.model_class):
            raise ValueError(
                f"model {model} is not of type {self.model_config.model_class} as specified in model_config.model_class"
            )

        self.model = model or self.model_config.model_class(**self.model_config.model_kwargs)

        self.add_event_handler(Events.STARTED, self.setup)
        for callback in model_zoo_entry.predictor_callbacks:
            self.add_event_handler(callback.event, callback.function)

    @staticmethod
    def setup(predictor: "Predictor"):
        predictor.model.eval()

    @staticmethod
    def predict_batch(predictor: "Predictor", batch) -> torch.Tensor:
        print("predict")
        return predictor.model(batch)
