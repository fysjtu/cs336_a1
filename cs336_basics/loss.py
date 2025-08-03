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


class LR_Cosine_Schedule(object):
    def __init__(self,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int
    ):
        self.max_lr = max_learning_rate
        self.min_lr = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def get_lr_at_it(self, it):
        current_lr = 0
        if it <= self.warmup_iters:
            current_lr = it/self.warmup_iters * self.max_lr
        elif it < self.cosine_cycle_iters:
            degree = (it-self.warmup_iters)  / (self.cosine_cycle_iters-self.warmup_iters) * math.pi
            current_lr = math.cos(degree) * (self.max_lr - self.min_lr) / 2 + (self.max_lr + self.min_lr) / 2
        else:
            current_lr = self.min_lr
        return current_lr




class CustomOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, **kwargs):
        defaults = dict(lr=lr, **kwargs)
        super(CustomOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """执行一步优化"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 在这里实现具体的参数更新逻辑
                grad = p.grad.data
                # 更新参数
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss
