from ignite.engine import Events, Engine

from protozoo.tiktorch_config_keys import ModelZooEntry


class Trainer(Engine):
    def __init__(self, model_zoo_entry: ModelZooEntry):
        model_config = model_zoo_entry.model_config
        optimizer_config = model_zoo_entry.optimizer_config
        loss_config = model_zoo_entry.loss_config

        if model_config.pretrained_source is not None:
            raise NotImplementedError("model_zoo_entry.model_config.pretrained_source")

        def training_step(trainer: Engine, batch) -> float:
            print("STEP")
            ipt, tgt = batch
            trainer.state.optimizer.zero_grad()
            pred = trainer.state.model(ipt)
            loss = trainer.state.loss_fn(pred, tgt)
            loss.backward()
            trainer.state.optimizer.step()
            return loss.item()

        super().__init__(training_step)

        @self.on(Events.STARTED)
        def setup(trainer: Engine):
            model = model_config.model_class(**model_config.model_kwargs)
            trainer.state.optimizer = optimizer_config.optimizer_class(
                model.parameters(), **optimizer_config.optimizer_kwargs
            )
            trainer.state.loss_fn = loss_config.loss_class(model.parameters(), **loss_config.loss_kwargs)
            model.train()
            trainer.state.model = model

        for callback in model_zoo_entry.trainer_callbacks:
            self.add_event_handler(callback.event, callback.function)
