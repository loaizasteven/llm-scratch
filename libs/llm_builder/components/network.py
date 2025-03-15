from torch import nn

class FeedFroward(nn.Module):
    """FF Network for Transformer Models (PyTorch)
    This class implements a feedforward network, which is a core component of transformer models.
    It consists of two linear layers with GELU activation in between. The hidden layer expands the
    dimention by 4 and then contracts it back to the original dimention.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)
    