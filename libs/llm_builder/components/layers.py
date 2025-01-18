from pydantic import BaseModel, Field

import torch
from torch import nn


class SelfAttention(nn.Module):
    """
    Self-Attention Layer for Transformer Models (PyTorch)
    This class implements a self-attention mechanism, which is a core component of transformer models. 
    It computes attention scores between input sequences to capture dependencies and relationships.
    Attributes:
        d_in (int): Dimensionality of the input features.
        d_out (int): Dimensionality of the output features.
        qkv_bias (bool): Whether to include a bias term in the linear projections for query, key, and value.
    Methods:
        forward(x):
            Computes the self-attention mechanism on the input tensor `x`.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_in).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_length, d_out) after applying self-attention.
    """
 
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
