import torch.nn as nn
import torch
import einops


class RoPE(nn.Module):
    def __init__(self, theta=10000):
        super(RoPE, self).__init__()
        self.theta = theta

    def rotate_half(self, x):
        x = einops.rearrange(x, "... (n r) -> ... n r", r=2)
        x1, x2 = x.unbind(-1)  # (..., d/2) each
        x = torch.stack(-x2, x1, dim=-1)  # (..., d/2, 2)
        return einops.rearrange(x, "... n r -> ... (n r)")

    def forward(self, q, k):
        # (bs, n_heads, seq_len, head_dim)
        head_dim = q.size(-1)
        seq_idxs = torch.arange(q.size(-2), device=q.device, dtype=q.dtype)
        thetas = 1 / self.theta ** (
            torch.arange(0, head_dim, 2, device=q.device, dtype=q.dtype) / head_dim
        )  # d/2 for one single pos
        thetas = torch.einsum("..., i -> ... i", seq_idxs, thetas)  # (seq_len, d/2)
        thetas = einops.repeat(thetas, "... i -> ... (n 2)")  # (seq_len, d)

        q_rot = q * torch.cos(thetas) + self.rotate_half(q) * torch.sin(thetas)
        k_rot = k * torch.cos(thetas) + self.rotate_half(k) * torch.sin(thetas)
        return q_rot, k_rot
