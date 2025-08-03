from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
import traceback
from cs336_basics import utils


def swish_activation(x):
    return x * torch.sigmoid(x)


def softmax(x, dim=-1):
    x_shifted = x - torch.max(x, dim=dim, keepdim=True).values
    out = torch.exp(x_shifted) / torch.sum(torch.exp(x_shifted), dim=dim, keepdim=True)
    return out


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, weights=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if weights is not None:
            self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
            self.weights_init(weights)
        else:
            self.weight = nn.Parameter(utils.xavier_init((in_dim, out_dim)))
    
    def weights_init(self, weights):
        try:
            self.weight.data.copy_(weights)
        except:
            error_msg = f"Linear_Layer weights initialization error: {traceback.format_exc()}"
            raise RuntimeError(error_msg)

    def forward(self, x):
        return x @ self.weight.T


class Embd(nn.Module):
    def __init__(self, vocab_size, d_model, weights=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        if weights is not None:
            self.weight = nn.Parameter(torch.zeros(vocab_size, d_model)) # [vocab, d_model]
            self.weight.data.copy_(weights)
        else:
            self.weight = nn.Parameter(utils.normal_init([vocab_size, d_model])) # [vocab, d_model]

    def forward(self, x):
        return self.weight[x]


class RoPE_Embd(nn.Module):
    def __init__(self, d_k, theta, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.theta = theta
        # constant no need backward
        self.cos_freqs = torch.empty(max_seq_len, d_k//2)
        self.sin_freqs = torch.empty(max_seq_len, d_k//2)
        self.rope_init()
    
    def rope_init(self):
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, dtype=torch.float32) / self.d_k))  # [d_k/2]
        positions = torch.arange(0, self.max_seq_len, dtype=torch.float32)
        pos_freqs = positions.unsqueeze(-1).to(torch.float32) * freqs # [maxlen, d_k/2]
        self.cos_freqs.copy_(torch.cos(pos_freqs))  # [maxlen, d_k/2]
        self.sin_freqs.copy_(torch.sin(pos_freqs))

    def forward(self, x, pos=None):
        slen = x.shape[-2]
        x_even = x[..., 0::2]  # 偶数位置
        x_odd = x[..., 1::2]   # 奇数位置
        if pos is None:
            cos_freqs = self.cos_freqs[:slen, :]
            sin_freqs = self.sin_freqs[:slen, :]
        else:
            cos_freqs = self.cos_freqs[pos, :]
            sin_freqs = self.sin_freqs[pos, :]

        rotated_even = x_even * cos_freqs - x_odd * sin_freqs
        rotated_odd = x_even * sin_freqs + x_odd * cos_freqs

        result = torch.zeros_like(x)
        result[..., 0::2] = rotated_even
        result[..., 1::2] = rotated_odd
        return result


# to keep the parameters not too large need scaling hidden_dim
class SwiGLU_Layer(nn.Module):
    def __init__(self, in_dim, hidden_dim, weights_list = None):
        super().__init__()
        
        if weights_list is not None:
            w1_weight = weights_list[0]
            w2_weight = weights_list[1]
            w3_weight = weights_list[2]
        else:
            w1_weight, w2_weight, w3_weight = None, None, None
            
        self.layer1 = Linear(in_dim, hidden_dim, w1_weight)
        self.layer2 = Linear(hidden_dim, in_dim, w2_weight)
        self.layer3 = Linear(in_dim, hidden_dim, w3_weight)

    def forward(self, x):
        x1 = self.layer1(x)
        x3 =  self.layer3(x)
        gate_output = swish_activation(x1) * x3
        output = self.layer2(gate_output)
        return output

class RMSNorm_Layer(nn.Module):
    def __init__(self, d_model, eps, weights = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.affine = nn.Parameter(utils.normal_init([d_model]))
        if weights is not None:
            self.weights_init(weights)

    def weights_init(self, weights):
        try:
            self.affine.data.copy_(weights)
        except:
            error_msg = f"RMSNorm_Layer weights initialization error: {traceback.format_exc()}"
            raise RuntimeError(error_msg)

    def forward(self, x):
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True)+self.eps)
        return self.affine * x / rms