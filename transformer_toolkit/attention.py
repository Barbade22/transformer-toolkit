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

    def forward(self, x, mask=None, past_kv=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        def split(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)

        if isinstance(self.pos_enc, RoPE):
            offset = past_kv[0].shape[2] if past_kv is not None else 0
            q, k = self.pos_enc.rotate(q, k, offset=offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v)

        is_causal = mask is None and past_kv is None
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask  = mask,
            is_causal  = is_causal,
            dropout_p  = 0.0,
        )
        return self.out(out.transpose(1, 2).reshape(B, -1, C)), present_kv


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

    def forward(self, x, mask=None, past_kv=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).split(self.n_kv_heads * self.head_dim, dim=-1)
        def split_kv(t): return t.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k, v = split_kv(k), split_kv(v)

        if isinstance(self.pos_enc, RoPE):
            offset = past_kv[0].shape[2] if past_kv is not None else 0
            q, k = self.pos_enc.rotate(q, k, offset=offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v)

        r = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)

        is_causal = mask is None and past_kv is None
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask  = mask,
            is_causal  = is_causal,
            dropout_p  = 0.0,
        )
        return self.out(out.transpose(1, 2).reshape(B, -1, C)), present_kv


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

    def forward(self, x, mask=None, past_kv=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, 1, self.head_dim).transpose(1, 2)

        if isinstance(self.pos_enc, RoPE):
            offset = past_kv[0].shape[2] if past_kv is not None else 0
            q, k = self.pos_enc.rotate(q, k, offset=offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v)

        # expand to match q heads after cache concat
        k = k.expand(B, self.n_heads, k.shape[2], self.head_dim).contiguous()
        v = v.expand(B, self.n_heads, v.shape[2], self.head_dim).contiguous()

        is_causal = mask is None and past_kv is None
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask  = mask,
            is_causal  = is_causal,
            dropout_p  = 0.0,
        )
        return self.out(out.transpose(1, 2).reshape(B, -1, C)), present_kv


class FlashAttention(nn.Module):
    """
    Flash Attention — same result as MHA, far less memory.
    Uses torch.nn.functional.scaled_dot_product_attention (torch >= 2.0).
    Kept as a separate class for clarity — identical to MHA in implementation
    but signals intent: use this when memory is the bottleneck.
    Note: KV cache disables the flash memory savings (cache materialises K/V),
    but correctness is preserved.
    """
    def __init__(self, dim: int, n_heads: int, pos_enc=None):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.qkv      = nn.Linear(dim, 3 * dim, bias=False)
        self.out      = nn.Linear(dim, dim, bias=False)
        self.pos_enc  = pos_enc

    def forward(self, x, mask=None, past_kv=None, causal=True):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        def split(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)

        if isinstance(self.pos_enc, RoPE):
            offset = past_kv[0].shape[2] if past_kv is not None else 0
            q, k = self.pos_enc.rotate(q, k, offset=offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v)

        is_causal = mask is None and past_kv is None and causal
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask  = mask,
            is_causal  = is_causal,
            dropout_p  = 0.0,
        )
        return self.out(out.transpose(1, 2).reshape(B, -1, C)), present_kv


class MLAttention(nn.Module):
    """
    Multi-Head Latent Attention (DeepSeek-V2/V3).
    Compresses k/v into a small latent vector to reduce KV cache at inference.
    Cache stores the latent vector (not full K/V) — that's the whole point.
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

    def forward(self, x, mask=None, past_kv=None):
        B, T, C = x.shape
        q      = self.q(x)
        latent = self.kv_down(x)   # [B, T, latent_dim] — this is what we cache

        if past_kv is not None:
            # past_kv is the accumulated latent cache [B, S, latent_dim]
            latent_full = torch.cat([past_kv, latent], dim=1)
        else:
            latent_full = latent

        present_kv = latent_full   # store latent, not K/V

        k = self.k_up(latent_full)
        v = self.v_up(latent_full)

        def split(t, seq): return t.view(B, seq, self.n_heads, self.head_dim).transpose(1, 2)
        q   = split(q,           T)
        k   = split(k, latent_full.shape[1])
        v   = split(v, latent_full.shape[1])

        if isinstance(self.pos_enc, RoPE):
            offset = past_kv.shape[1] if past_kv is not None else 0
            q, k = self.pos_enc.rotate(q, k, offset=offset)

        is_causal = mask is None and past_kv is None
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask  = mask,
            is_causal  = is_causal,
            dropout_p  = 0.0,
        )
        return self.out(out.transpose(1, 2).reshape(B, -1, C)), present_kv