from typing import Tuple

import torch

from protozoo.config_pytorch import TorchMiniBatchOfInput


def default(mini_batch: TorchMiniBatchOfInput) -> Tuple[torch.Tensor]:
    """
    Extract the input torch.Tensor from a mini_batch of type `TorchMiniBatchOfInput` (or inherited thereof)
    """

    return (mini_batch.input_tensor,)
