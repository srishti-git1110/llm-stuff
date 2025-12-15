from architectures.pos_embed.rope import RoPE
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super(CausalSelfAttention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope = RoPE()

        self.qkv_proj = nn.Linear(dim, 3*dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, sl, _ = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bs, sl, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, sl, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, sl, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        attn_scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        mask = torch.tril(torch.ones(sl, sl, device=x.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf")) # prevent lookin ahead
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = attn_probs @ v
        attn_out = attn_out.transpose(1, 2).contiguous().view(bs, sl, self.dim)

        return self.wo(attn_out)
