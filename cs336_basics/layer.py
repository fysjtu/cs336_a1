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
        swish_x = x1 * torch.sigmoid(x1)
        gate_output = swish_x * x3
        output = self.layer2(gate_output)
        return output

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError



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