dependencies = ["torch", "inferno"]

from DUNet import DUNet2D

def dunet2d(pretrained=False, **kwargs):
    assert not pretrained
    model = DUNet2D(**kwargs)
    return model
