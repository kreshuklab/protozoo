from typing import Tuple

import torch

from protozoo.config_pytorch import TorchMiniBatchOfInputTarget


def default(model_output: torch.Tensor, mini_batch: TorchMiniBatchOfInputTarget) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines the model output with the mini batch target.
    """

    return model_output, mini_batch.target_tensor
