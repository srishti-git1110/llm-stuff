import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, sl, dim)
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.w * (x / norm)