from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import einsum, rearrange
from cs336_basics.layer import *



def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:


    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    
    output = torch.matmul(attention_weights, V) 
    
    return output






class MultiheadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False, use_casual_mask: bool = True,
                    rope_param:tuple = None, weights_list:list = None):
        super().__init__()
        self.d_model = d_model
        self.dk = d_model // num_heads
        self.dv = d_model // num_heads
        self.num_heads = num_heads
        self.use_casual_mask = use_casual_mask
        if use_rope:
            (self.max_len, self.theta) = rope_param
            self.rope_embd = RoPE_Embd(self.dk, self.theta, self.max_len)
        else:
            self.rope_embd = None
        if weights_list is not None and all(w is not None for w in weights_list[:3]):
            self.qkv_weight = torch.concat(weights_list[:3])
            self.o_weight = weights_list[3] if len(weights_list) > 3 and weights_list[3] is not None else None
        else:
            self.qkv_weight = None
            self.o_weight = None
        print('self.qkv',self.qkv_weight, self.o_weight)
        self.qkv_proj = Linear(d_model, 3*self.d_model, self.qkv_weight)
        self.o_proj = Linear(d_model, d_model, self.o_weight)


    def forward(self, x: torch.Tensor, pos: torch.Tensor = None):
        slen = x.shape[-2]
        b = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, -1)
    
        qm = rearrange(q, "b slen (h dk) -> b h slen dk", h=self.num_heads)
        km = rearrange(k, "b slen (h dk) -> b h slen dk", h=self.num_heads)
        vm = rearrange(v, "b slen (h dv) -> b h slen dv", h=self.num_heads)

        if self.rope_embd is not None:
            qm = self.rope_embd(qm, pos)
            km = self.rope_embd(km, pos)
            # not need rope for v!
            # vm = self.rope_embd(vm, pos)
            
        casual_mask = None
        if self.use_casual_mask:
            casual_mask = torch.triu(torch.ones(slen, slen), diagonal=1).bool()
            casual_mask = ~casual_mask[None, None, :, :] 
        output = scaled_dot_product_attention(qm, km, vm, casual_mask)
        output = rearrange(
            output, "... h seq_len d_head ->  ... seq_len (h d_head)"
        )
        return self.o_proj(output)


