# positional_encodings.py

import torch
import torch.nn as nn
import math


# ─── stream encodings (applied to x before the transformer blocks) ────────────

class SinusoidalPE(nn.Module):
    """
    Added to the residual stream once before the first block.
    Attention modules are unaware of it.
    No learnable parameters.
    """
    def __init__(self, dim: int, max_seq: int = 4096):
        super().__init__()
        pe  = torch.zeros(max_seq, dim)
        pos = torch.arange(max_seq).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, dim]
        return x + self.pe[:x.shape[1]]


class LearnedPE(nn.Module):
    """
    Added to the residual stream once before the first block.
    Attention modules are unaware of it.
    Learnable embedding table.
    """
    def __init__(self, dim: int, max_seq: int = 4096):
        super().__init__()
        self.pe = nn.Embedding(max_seq, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.shape[1], device=x.device)
        return x + self.pe(pos)


# ─── attention encodings (applied inside each attention module) ────────────────

class RoPE(nn.Module):
    """
    Applied inside attention to q and k AFTER head-splitting.
    q, k shape expected: [B, n_heads, T, head_dim]

    Instantiated once in Transformer.__init__() with head_dim
    (= dim // n_heads) and passed into every attention module so
    all layers share the same frequency table and cache.

    Never called on the residual stream x.
    """
    def __init__(self, head_dim: int, max_seq: int = 4096, base: int = 10000):
        super().__init__()
        self._head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int):
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)            # [seq_len, head_dim//2]
        emb   = torch.cat([freqs, freqs], dim=-1)        # [seq_len, head_dim]
        # [1, 1, seq_len, head_dim] — broadcasts over batch and n_heads
        self.register_buffer("cos_cache", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None], persistent=False)

    def rotate(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """
        q, k: [B, n_heads, T, head_dim]
        offset: number of tokens already in the KV cache (0 during training
                or prefill, >0 during cached decode steps).
        Returns rotated (q, k) with identical shape.
        """
        T   = q.shape[2]
        end = offset + T
        if end > self.cos_cache.shape[2]:
            self._build_cache(end)
        # cast to q's dtype — cache is kept float32 for precision but q/k
        # may be bfloat16/float16 under mixed precision training
        cos = self.cos_cache[:, :, offset:end, :].to(dtype=q.dtype)
        sin = self.sin_cache[:, :, offset:end, :].to(dtype=q.dtype)
        return (
            q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin,
        )


class ALiBi(nn.Module):
    """
    Applied inside attention by adding a bias to scores BEFORE softmax.
    scores shape: [B, n_heads, T, T]

    Instantiated once in Transformer.__init__() and passed into every
    attention module. Transformer.forward() calls get_bias() once per
    forward pass and passes the result as the mask argument to each block.

    Never modifies the residual stream x.
    """
    def __init__(self, n_heads: int):
        super().__init__()
        slopes = 2 ** (-8 * torch.arange(1, n_heads + 1) / n_heads)
        self.register_buffer("slopes", slopes)

    def get_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns additive float bias [n_heads, T, T].
        Past tokens get a small negative penalty growing with distance.
        Future tokens get -inf (causal mask).
        """
        pos  = torch.arange(seq_len, device=device)
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)             # [T, T]
        bias = self.slopes[:, None, None] * dist.unsqueeze(0)  # [n_heads, T, T]
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return bias.masked_fill(causal.unsqueeze(0), float('-inf'))


# ─── shared utility ───────────────────────────────────────────────────────────

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)