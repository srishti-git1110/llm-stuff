from architectures.decoder_only.causal_self_attn import CausalSelfAttention
from architectures.decoder_only.ff import FFNN
from architectures.decoder_only.rms_norm import RMSNorm
from architectures.pos_embed.rope import RoPE
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffnn_hid_dim: int, rms_eps: float = 1e-08):
        super(DecoderBlock, self).__init__()
        self.norm = RMSNorm(dim, eps=rms_eps)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ffnn = FFNN(dim, ffnn_hid_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x))
        x = x + self.ffnn(self.norm(x))
        return x
    

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_sl: int, n_layers: int, dim: int, n_heads: int, ffnn_hid_dim: int, rms_eps: float = 1e-08):
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        self.layers = nn.ModuleList([
            DecoderBlock(dim, n_heads, ffnn_hid_dim, rms_eps) 
            for _ in range(n_layers) 
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)
