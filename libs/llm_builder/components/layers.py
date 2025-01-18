from pydantic import BaseModel, Field

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, dqv_bias: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.qkv_bias = dqv_bias
        self.W_query = nn.Linear(self.d_in, self.d_out, bias=self.qkv_bias)
        self.W_key = nn.Linear(self.d_in, self.d_out, bias=self.qkv_bias)
        self.W_value = nn.Linear(self.d_in, self.d_out, bias=self.qkv_bias)
        
    def forward(self, x):
        key = self.W_key(x)
        query = self.W_query(x)
        value = self.W_value(x)

        attn_score = query @ key.T
        attn_weights = torch.softmax(
            attn_score / key.shape[-1]**0.5, dim=-1
        )

        context_vector = attn_weights @ value

        return context_vector
