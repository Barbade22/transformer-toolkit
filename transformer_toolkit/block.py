# block.py

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from .feed_forward  import FFN
from .normalization import LayerNorm
from .attention     import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block: norm → attn → residual → norm → ffn → residual.
    Swap any component via attn=, ffn=, norm=.

    forward() always returns (x, aux_loss, present_kv).
    aux_loss is non-zero only when an MoE FFN is used.
    present_kv is None when use_kv_cache=False (training default).

    Gradient checkpointing is supported: set use_checkpoint=True to
    trade compute for memory. KV cache and gradient checkpointing are
    mutually exclusive — checkpointing is a training-only feature.
    """
    def __init__(
        self,
        dim:             int,
        n_heads:         int,
        hidden:          int,
        norm             = None,
        attn             = None,
        ffn              = None,
        dropout:         float = 0.0,
        use_checkpoint:  bool  = False,
    ):
        super().__init__()
        if norm is not None:
            norm_cls    = type(norm)
            norm_kwargs = _norm_kwargs(norm)
            self.norm1  = norm_cls(**norm_kwargs)
            self.norm2  = norm_cls(**norm_kwargs)
        else:
            self.norm1 = LayerNorm(dim)
            self.norm2 = LayerNorm(dim)

        self.attn           = attn or MultiHeadAttention(dim, n_heads)
        self.ffn            = ffn  or FFN(dim, hidden)
        self.drop           = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint
        self._is_moe        = hasattr(self.ffn, "aux_weight")

    def forward(self, x: torch.Tensor, mask=None, past_kv=None, use_kv_cache: bool = False):
        if self.use_checkpoint:
            # gradient checkpointing — no KV cache (training only)
            if mask is None:
                x, aux_loss = torch_checkpoint.checkpoint(
                    self._forward_no_cache, x, use_reentrant=False
                )
            else:
                x, aux_loss = torch_checkpoint.checkpoint(
                    self._forward_no_cache, x, mask, use_reentrant=False
                )
            return x, aux_loss, None

        return self._forward(x, mask=mask, past_kv=past_kv, use_kv_cache=use_kv_cache)

    def _forward(self, x: torch.Tensor, mask=None, past_kv=None, use_kv_cache: bool = False):
        attn_out, present_kv = self.attn(self.norm1(x), mask=mask, past_kv=past_kv)
        x = x + self.drop(attn_out)

        if self._is_moe:
            ffn_out, aux_loss = self.ffn(self.norm2(x))
        else:
            ffn_out  = self.ffn(self.norm2(x))
            aux_loss = 0.0

        x = x + self.drop(ffn_out)
        return x, aux_loss, present_kv if use_kv_cache else None

    def _forward_no_cache(self, x: torch.Tensor, mask=None):
        """Used by gradient checkpointing — no past_kv threading."""
        attn_out, _ = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop(attn_out)
        if self._is_moe:
            ffn_out, aux_loss = self.ffn(self.norm2(x))
        else:
            ffn_out  = self.ffn(self.norm2(x))
            aux_loss = 0.0
        x = x + self.drop(ffn_out)
        return x, aux_loss


# ─── helper ───────────────────────────────────────────────────────────────────

def _norm_kwargs(norm: nn.Module) -> dict:
    """Extract constructor kwargs from a norm instance so we can re-instantiate it."""
    if hasattr(norm, "w") and hasattr(norm, "eps"):
        dim = norm.w.shape[0]
        return {"dim": dim, "eps": norm.eps}
    if hasattr(norm, "normalized_shape"):
        return {"dim": norm.normalized_shape[0],
                "eps": getattr(norm, "eps", 1e-5)}
    raise TypeError(
        f"Cannot extract constructor kwargs from norm type {type(norm).__name__}. "
        "Pass dim= and eps= explicitly or subclass to expose them."
    )