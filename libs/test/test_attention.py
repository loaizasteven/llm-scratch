from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
from libs.llm_builder.components.layers import SelfAttention, CausalAttention, MultiHeadAttentionWrapper

x=torch.tensor([[1., 2.], [2., 3.], [3., 4.]])

def test_attention():
    attention = SelfAttention(d_in=2, d_out=2)
    output = attention.forward(x=x)
    assert output.shape == (3, 2)

def test_causal_attention():
    x_batch = torch.stack([x,x], dim=0)

    causal_attention = CausalAttention(d_in=2, d_out=2, context_length=x_batch.shape[1], dropout=0.1)
    output = causal_attention.forward(x=x_batch)
    assert output.shape[0] == 2 # batch size

def test_multihead_attention():
    x_batch = torch.stack([x,x], dim=0)

    multihead_attention = MultiHeadAttentionWrapper(d_in=2, d_out=2, num_heads=2, context_length=x_batch.shape[1], dropout=0.9)
    output = multihead_attention.forward(x=x_batch)

    assert output.shape[2] == 4 # dim_out * num_heads
