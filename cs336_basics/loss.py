from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
import traceback
import math
import numpy.typing as npt
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import math
from einops import einsum, rearrange
from cs336_basics.layer import *



def ce_loss(model_output: Float[Tensor, " batch_size vocab_size"], labels: Int[Tensor, " batch_size"]):
    x_max = torch.max(model_output, dim=-1, keepdim=True).values
    x_shifted = model_output - x_max
    log_sum_exp = torch.log(torch.sum(torch.exp(x_shifted), dim=-1, keepdim=True)) # [b, 1]
    log_logits = x_shifted - log_sum_exp # [b, vocab] 
    neg_log_likehood = -log_logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze() # [b]
    return neg_log_likehood.mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    l2_norm = 0
    for param in parameters:
        if param.grad is not None:
            l2_norm += torch.sum(torch.square(param.grad))
    l2_norm = torch.sqrt(l2_norm)
    if l2_norm > max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad = param.grad / (l2_norm / max_l2_norm)
    return parameters

