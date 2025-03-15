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


class CausalAttention(SelfAttention):
    """
    Causal Attention Layer for Transformer Models (PyTorch)
    This class implements a causal attention mechanism, which is a core component of transformer models. 
    It computes attention scores between input sequences to capture dependencies and relationships, while preventing
    information leakage from future tokens. Additionally, it includes a dropout layer to regularize the attention weights.
    Attributes:
        d_in (int): Dimensionality of the input features.
        d_out (int): Dimensionality of the output features.
        context_length (int): Length of the context for the attention mechanism.
        dropout (float): Dropout rate applied to the attention weights.
        qkv_bias (bool): Whether to include a bias term in the linear projections for query, key, and value.
    Methods:
        forward(x):
            Computes the causal attention mechanism on the input tensor `x`.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_in).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_length, d_out) after applying causal attention.
    """
    
    def __init__(self, d_in: int, d_out: int, context_length:int, dropout:float, qkv_bias: bool = True):
        super().__init__(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Matrix of diagnoal 1s, to be used as a mask
        # registered as a buffer to be moved to the device with the model
        # this is a constant tensor that does not require gradients
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        key = self.W_key(x)
        query = self.W_query(x)
        value = self.W_value(x)

        # Transpose dimentions 1 and 2, keeping the batch dimension at the first index
        attn_score = query @ key.transpose(1,2)
        # Apply softmax to "-inf" values results in 0 and applies a mask to the future tokens
        # In Pytorch, operations with trailing underscores modify the tensor in-place
        attn_score = attn_score.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_score / key.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ value
        return context_vector


class MultiHeadAttentionWrapper(nn.Module):
    """
    Stacked Multi-Head Attention Layer for Transformer Models (PyTorch)
    This class implements a multi-head attention mechanism, which is a core component of transformer models.
    It computes multiple attention heads sequentially and concatenates the results to provide a richer representation.
    """
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """Similar to MultiHeadAttentionWrapper but with a single class
    Additionally, d_out is the multihead output dimension instead of the single head output dimension as defined in 
    previous classes.
    """
    def __init__(self, d_in: int, d_out: int, context_length:int, num_heads:int, dropout:float, qkv_bias: bool = True):
        super().__init__(d_in, d_out, qkv_bias)
        assert(d_out % num_heads == 0), \
            "Number of heads must be a factor of the output dimension"
        
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_in = d_in
        self.d_out = d_out
        self.qkv_bias = dqv_bias
        self.W_query = nn.Linear(self.d_in, self.d_out, bias=self.qkv_bias)
        self.W_key = nn.Linear(self.d_in, self.d_out, bias=self.qkv_bias)
        self.W_value = nn.Linear(self.d_in, self.d_out, bias=self.qkv_bias)
        self.out_proj = nn.Linear(self.d_out, self.d_out)
        # Matrix of diagnoal 1s, to be used as a mask
        # registered as a buffer to be moved to the device with the model
        # this is a constant tensor that does not require gradients
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # shape b, num_tokens, d_out
        key = self.W_key(x)
        query = self.W_query(x)
        value = self.W_value(x)

        # d_out -> (num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose from `b, num_tokens, num_heads, head_dim` to `b, num_heads, num_tokens, head_dim`
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)


        attn_score = query @ key.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Apply softmax to "-inf" values results in 0 and applies a mask to the future tokens
        # In Pytorch, operations with trailing underscores modify the tensor in-place
        attn_score = attn_score.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_score / key.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ value).transpose(1,2)
        # Place tensor in contiguous memory blocks for reshaping
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)
        return context_vector

class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift