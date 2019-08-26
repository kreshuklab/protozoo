from torch.utils.data import Dataset
from skimage.data import astronaut


class MockData(Dataset):
    def __getitem__(self, key: int):
        assert key < 3
        ipt = astronaut()
        return ipt, ipt

    def __len__(self):
        return 3
