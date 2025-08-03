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


class TransformerModel(nn.Module):
    def __init__(self,
        vocab_size: int,
        num_layers: int,
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
        weights_dict: dict[str, Tensor] = None
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.weights_extract(weights_dict)
        self.embd = Embd(vocab_size, d_model, weights=self.trfm_model_weights_dict['embd_weights'])
        self.transform_layers = nn.ModuleList(
            TransformerBlock(
                d_model = d_model, 
                num_heads = num_heads, 
                d_ff = d_ff, 
                max_seq_len = max_seq_len,
                theta = theta,
                weights_dict = self.trfm_model_weights_dict['trfm_weights_list'][i]
            ) for i in range(num_layers)
        )
        self.final_norm = RMSNorm_Layer(d_model, eps, weights=self.trfm_model_weights_dict['ln_final'])
        self.final_linear = Linear(d_model, vocab_size, weights = self.trfm_model_weights_dict['final_linear'])

    def weights_extract(self, weights_dict):
        self.trfm_model_weights_dict ={
            'trfm_weights_list':[
                {
                    'attn.q_proj.weight': weights_dict.get(('layers.{}.attn.q_proj.weight'.format(str(i))), None),
                    'attn.k_proj.weight': weights_dict.get(('layers.{}.attn.k_proj.weight'.format(str(i))), None),
                    'attn.v_proj.weight': weights_dict.get(('layers.{}.attn.v_proj.weight'.format(str(i))), None),
                    'attn.o_proj.weight': weights_dict.get(('layers.{}.attn.o_proj.weight'.format(str(i))), None),
                    'ffn.w1.weight': weights_dict.get(('layers.{}.ffn.w1.weight'.format(str(i))), None),
                    'ffn.w2.weight': weights_dict.get(('layers.{}.ffn.w2.weight'.format(str(i))), None),
                    'ffn.w3.weight': weights_dict.get(('layers.{}.ffn.w3.weight'.format(str(i))), None),
                    'ln1.weight': weights_dict.get(('layers.{}.ln1.weight'.format(str(i))), None),
                    'ln2.weight': weights_dict.get(('layers.{}.ln2.weight'.format(str(i))), None)
                }   for i in range(self.num_layers)
            ],
            'embd_weights': weights_dict.get('token_embeddings.weight', None),
            'ln_final': weights_dict.get('ln_final.weight', None),
            'final_linear': weights_dict.get('lm_head.weight', None),
        }

    def forward(self, indices):
        x =  self.embd(indices) # [b, slen, dim]
        for i in range(self.num_layers): 
            x = self.transform_layers[i](x) # [b, slen, dim]
        x = self.final_norm(x)
        x = self.final_linear(x) # [b, slen, vocab_size]
        return x
    
    def predict(self, x):
        x = self.forward(x)
        return x
