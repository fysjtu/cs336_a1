from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
import traceback

import numpy.typing as npt
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import einsum, rearrange
from cs336_basics.layer import *
from cs336_basics.attention import *





class TransformerBlock(nn.Module):
    def __init__(self,
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
        weights_dict: dict[str, Tensor] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.weights_extract(weights_dict)

        self.mha = MultiheadSelfAttention(d_model, num_heads, use_rope=True, rope_param=(max_seq_len, theta), weights_list=self.mha_weights_list)
        self.ffn = SwiGLU_Layer(d_model, d_ff, weights_list=self.ffn_weights_list)
        self.norm1 = RMSNorm_Layer(d_model, eps, weights=self.norm1_weight)
        self.norm2 = RMSNorm_Layer(d_model, eps, weights=self.norm2_weight)
    
    def weights_extract(self, weights_dict: dict[str, Tensor]):
        try:
            if weights_dict is not None:
                self.mha_weights_list = [
                    weights_dict.get('attn.q_proj.weight', None),
                    weights_dict.get('attn.k_proj.weight', None),
                    weights_dict.get('attn.v_proj.weight', None),
                    weights_dict.get('attn.output_proj.weight', None)  # 注意：应该是 output_proj
                ]

                self.ffn_weights_list = [
                    weights_dict.get('ffn.w1.weight', None),
                    weights_dict.get('ffn.w2.weight', None),
                    weights_dict.get('ffn.w3.weight', None),
                ]

                self.norm1_weight = weights_dict.get('ln1.weight', None)
                self.norm2_weight = weights_dict.get('ln2.weight', None)
            else:
                self.mha_weights_list = [None, None, None, None]
                self.ffn_weights_list = [None, None, None]
                self.norm1_weight = None
                self.norm2_weight = None
        except Exception as e:
            error_msg = f"Transformer weights initialization error: {traceback.format_exc()}"
            raise RuntimeError(error_msg)

    def forward(self, x):
        attn_out = x + self.mha(self.norm1(x))
        ffn_out = attn_out + self.ffn(self.norm2(attn_out))
        return ffn_out
