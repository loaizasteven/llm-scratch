from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
from libs.llm_builder.components.layers import SelfAttention

def test_attentionv1():
    attention = SelfAttention(d_in=2, d_out=2)
    output = attention.forward(x=torch.tensor([[1., 2.], [2., 3.], [3., 4.]]))
    assert output.shape == (3, 2)
    