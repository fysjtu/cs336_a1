from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math

def zero_init( vocab_size: int, d_model: int):
    return torch.zeros((vocab_size, d_model))

def xavier_init(tensor_shape) -> torch.Tensor:
    in_dim, out_dim = tensor_shape[-2], tensor_shape[-1]
    W = torch.empty(out_dim, in_dim)
    mean = 0
    std = np.sqrt(2 / (in_dim + out_dim))
    nn.init.trunc_normal_(W, mean, std, -3*std, 3*std)
    return W

def normal_init(tensor_shape) -> torch.Tensor:
    W = torch.empty(tensor_shape)
    mean = 0
    std = 1
    nn.init.trunc_normal_(W, mean, std, -3*std, 3*std)
    return W


if __name__ == '__main__':
    # 调用函数并检查属性
    my_tensor = normal_init((3, 4))
    my_tensor = xavier_init((3, 4))
    print(f"张量 W 的 requires_grad 属性是: {my_tensor.requires_grad}")