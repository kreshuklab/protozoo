from torch.utils.data import DataLoader

from protozoo.datasets import Fluo_N2DH_SIM


from exampleupload.modelzoo import get_entry
from protozoo.predictor import Predictor

if __name__ == "__main__":
    model_zoo_entry = get_entry()

    predictor = Predictor(model_zoo_entry)
    dl = DataLoader(Fluo_N2DH_SIM(load_target=False))

    predictor.run(dl)
    print("done")
