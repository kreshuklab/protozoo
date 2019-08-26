from ignite.engine import Events, Engine

from protozoo.tiktorch_config_keys import ModelZooEntry


def get_trainer(model_zoo_entry: ModelZooEntry) -> Engine:
    def training_step(trainer: Engine, batch) -> float:
        print("STEP")
        ipt, tgt = batch
        trainer.state.optimizer.zero_grad()
        pred = trainer.state.model(ipt)
        loss = trainer.state.loss_fn(pred, tgt)
        loss.backward()
        trainer.state.optimizer.step()
        return loss.item()

    trainer = Engine(training_step)

    @trainer.on(Events.STARTED)
    def training_setup(trainer: Engine):
        trainer.state.model = model_zoo_entry.model_config.model_class(**model_zoo_entry.model_config.model_kwargs)
        assert model_zoo_entry.model_config.pretrained_source is None
        trainer.state.optimizer = model_zoo_entry.optimizer_config.optimizer_class(
            trainer.state.model.parameters(), **model_zoo_entry.optimizer_config.optimizer_kwargs
        )
        trainer.state.loss_fn = model_zoo_entry.loss_config.loss_class(
            trainer.state.model.parameters(), **model_zoo_entry.loss_config.loss_kwargs
        )

    for callback in model_zoo_entry.trainer_callbacks:
        trainer.add_event_handler(callback.event, callback.function)

    return trainer
