from ignite.engine import Events, Engine

from protozoo.tiktorch_config_keys import ModelZooEntry


class Trainer(Engine):
    def __init__(self, model_zoo_entry: ModelZooEntry):
        self.model_config = model_zoo_entry.model_config
        self.optimizer_config = model_zoo_entry.optimizer_config
        self.loss_config = model_zoo_entry.loss_config

        if self.model_config.pretrained_source is not None:
            raise NotImplementedError("model_zoo_entry.model_config.pretrained_source")

        super().__init__(self.training_step)

        self.add_event_handler(Events.STARTED, self.setup)
        for callback in model_zoo_entry.trainer_callbacks:
            self.add_event_handler(callback.event, callback.function)

    @staticmethod
    def training_step(trainer: "Trainer", batch) -> float:
        print("STEP")
        ipt, tgt = batch
        trainer.state.optimizer.zero_grad()
        pred = trainer.model(ipt)
        loss = trainer.state.loss_fn(pred, tgt)
        loss.backward()
        trainer.state.optimizer.step()
        return loss.item()

    @staticmethod
    def setup(trainer: "Trainer"):
        trainer.model = trainer.model_config.model_class(**trainer.model_config.model_kwargs)
        trainer.state.optimizer = trainer.optimizer_config.optimizer_class(
            trainer.model.parameters(), **trainer.optimizer_config.optimizer_kwargs
        )
        trainer.state.loss_fn = trainer.loss_config.loss_class(
            trainer.model.parameters(), **trainer.loss_config.loss_kwargs
        )
        trainer.model.train()
