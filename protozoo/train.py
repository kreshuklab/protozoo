from torch.utils.data import DataLoader

from protozoo.datasets import Fluo_N2DH_SIM
from protozoo.trainer import Trainer

from exampleupload.modelzoo import get_entry

if __name__ == "__main__":
    model_zoo_entry = get_entry()

    trainer = Trainer(model_zoo_entry)
    dl = DataLoader(Fluo_N2DH_SIM())

    trainer.run(dl)
    print("done")
