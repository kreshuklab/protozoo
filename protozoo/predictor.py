import torch

from ignite.engine import Events, Engine
from typing import Optional

from protozoo.tiktorch_config_keys import ModelZooEntry


class Predictor(Engine):
    def __init__(self, model_zoo_entry: ModelZooEntry, model: Optional[torch.nn.Module] = None):
        if model_zoo_entry.model_config.pretrained_source is not None:
            raise NotImplementedError("model_zoo_entry.model_config.pretrained_source")

        model_config = model_zoo_entry.model_config

        def predict_batch(predictor: Engine, batch) -> torch.Tensor:
            print("predict")
            return predictor.state.model(batch)

        super().__init__(predict_batch)

        self.given_model = model
        if model is not None and not isinstance(model, model_config.model_class):
            raise ValueError(
                f"model {model} is not of type {model_config.model_class} as specified in model_config.model_class"
            )

        @self.on(Events.STARTED)
        def setup(predictor: Predictor):
            predictor.state.model = predictor.given_model or model_config.model_class(**model_config.model_kwargs)
            predictor.state.model.eval()

        for callback in model_zoo_entry.predictor_callbacks:
            self.add_event_handler(callback.event, callback.function)
