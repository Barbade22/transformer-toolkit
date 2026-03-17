# attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encodings import RoPE


class MultiHeadAttention(nn.Module):
    """Classic MHA. Used in original Transformer, BERT, GPT-2."""
    def __init__(self, dim: int, n_heads: int, pos_enc=None):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.qkv      = nn.Linear(dim, 3 * dim, bias=False)
        self.out      = nn.Linear(dim, dim, bias=False)
        self.pos_enc  = pos_enc

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        def split(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        if isinstance(self.pos_enc, RoPE):
            q, k = self.pos_enc.rotate(q, k)
        if mask is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                  is_causal=True, dropout_p=0.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                  is_causal=False, dropout_p=0.0)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


class GroupedQueryAttention(nn.Module):
    """Fewer k/v heads than q heads. Used in LLaMA 3, Mistral."""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, pos_enc=None):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = dim // n_heads
        self.q          = nn.Linear(dim, dim, bias=False)
        self.kv         = nn.Linear(dim, 2 * n_kv_heads * self.head_dim, bias=False)
        self.out        = nn.Linear(dim, dim, bias=False)
        self.pos_enc    = pos_enc

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).split(self.n_kv_heads * self.head_dim, dim=-1)
        def split_kv(t): return t.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k, v = split_kv(k), split_kv(v)
        if isinstance(self.pos_enc, RoPE):
            q, k = self.pos_enc.rotate(q, k)
        r = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)
        if mask is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                  is_causal=True, dropout_p=0.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                  is_causal=False, dropout_p=0.0)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


class MultiQueryAttention(nn.Module):
    """Single k/v head shared across all q heads. Used in Falcon, early Gemini."""
    def __init__(self, dim: int, n_heads: int, pos_enc=None):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.q        = nn.Linear(dim, dim, bias=False)
        self.k        = nn.Linear(dim, self.head_dim, bias=False)
        self.v        = nn.Linear(dim, self.head_dim, bias=False)
        self.out      = nn.Linear(dim, dim, bias=False)
        self.pos_enc  = pos_enc

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # single k/v head — expand to match q heads, contiguous for CUDA kernels
        k = self.k(x).view(B, T, 1, self.head_dim).transpose(1, 2) \
                      .expand(B, self.n_heads, T, self.head_dim).contiguous()
        v = self.v(x).view(B, T, 1, self.head_dim).transpose(1, 2) \
                      .expand(B, self.n_heads, T, self.head_dim).contiguous()
        if isinstance(self.pos_enc, RoPE):
            q, k = self.pos_enc.rotate(q, k)
        if mask is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                  is_causal=True, dropout_p=0.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                  is_causal=False, dropout_p=0.0)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


class FlashAttention(nn.Module):
    """
    Flash Attention — same result as MHA, far less memory.
    Uses torch.nn.functional.scaled_dot_product_attention (torch >= 2.0).
    Kept as a separate class for clarity — identical to MHA in implementation
    but signals intent: use this when memory is the bottleneck.
    """
    def __init__(self, dim: int, n_heads: int, pos_enc=None):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.qkv      = nn.Linear(dim, 3 * dim, bias=False)
        self.out      = nn.Linear(dim, dim, bias=False)
        self.pos_enc  = pos_enc

    def forward(self, x, mask=None, causal=True):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        def split(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        if isinstance(self.pos_enc, RoPE):
            q, k = self.pos_enc.rotate(q, k)
        if mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                  is_causal=False, dropout_p=0.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                  is_causal=causal, dropout_p=0.0)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


class MLAttention(nn.Module):
    """
    Multi-Head Latent Attention (DeepSeek-V2/V3).
    Compresses k/v into a small latent vector to reduce KV cache at inference.
    """
    def __init__(self, dim: int, n_heads: int, latent_dim: int, pos_enc=None):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.kv_down  = nn.Linear(dim, latent_dim, bias=False)
        self.k_up     = nn.Linear(latent_dim, dim, bias=False)
        self.v_up     = nn.Linear(latent_dim, dim, bias=False)
        self.q        = nn.Linear(dim, dim, bias=False)
        self.out      = nn.Linear(dim, dim, bias=False)
        self.pos_enc  = pos_enc

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q      = self.q(x)
        latent = self.kv_down(x)   # [B, T, latent_dim] — cached at inference
        k      = self.k_up(latent)
        v      = self.v_up(latent)
        def split(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        if isinstance(self.pos_enc, RoPE):
            q, k = self.pos_enc.rotate(q, k)
        if mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                  is_causal=False, dropout_p=0.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                  is_causal=True, dropout_p=0.0)
        return self.out(out.transpose(1, 2).reshape(B, T, C))