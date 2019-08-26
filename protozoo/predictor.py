import torch
from ignite.engine import Events, Engine

from protozoo.tiktorch_config_keys import ModelZooEntry


def get_predictor(model_zoo_entry: ModelZooEntry) -> Engine:
    def predict_batch(predictor: Engine, batch) -> torch.Tensor:
        print('predict')
        return predictor.state.model(batch)


    predictor = Engine(predict_batch)

    @predictor.on(Events.STARTED)
    def training_setup(predictor: Engine):
        predictor.state.model = model_zoo_entry.model_config.model_class(**model_zoo_entry.model_config.model_kwargs)
        assert model_zoo_entry.model_config.pretrained_source is None

    for callback in model_zoo_entry.predictor_callbacks:
        predictor.add_event_handler(callback.event, callback.function)

    return predictor
