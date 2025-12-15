import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
        self.activation = F.silu()
        self.fc3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.activation(self.fc1(x)) + self.fc2(x))
