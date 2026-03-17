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

    forward() always returns (x, aux_loss).
    aux_loss is non-zero only when an MoE FFN is used; the Transformer
    accumulates it across layers and adds it to the training loss.
    Non-MoE blocks always return aux_loss = tensor(0.0).

    Gradient checkpointing is supported: set use_checkpoint=True to
    trade compute for memory. Uses PyTorch's native checkpoint utility,
    NOT the HuggingFace gradient_checkpointing_enable() method which
    requires a HF model wrapper.
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
        # Always two independent norm instances — never share the same object.
        # The norm= argument is used only to determine the class and config;
        # we instantiate fresh copies so norm1 and norm2 have separate parameters
        # and accumulate gradients independently.
        if norm is not None:
            # Re-instantiate from the passed norm's class and constructor args
            # so callers can still inject RMSNorm/LayerNorm/etc. via norm=
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

    def forward(self, x: torch.Tensor, mask=None):
        if self.use_checkpoint:
            # torch.utils.checkpoint with use_reentrant=False cannot handle
            # None in the argument list — call _forward directly with mask=None
            # or pass the mask as a tensor argument
            if mask is None:
                return torch_checkpoint.checkpoint(
                    self._forward, x, use_reentrant=False
                )
            return torch_checkpoint.checkpoint(
                self._forward, x, mask, use_reentrant=False
            )
        return self._forward(x, mask)

    def _forward(self, x: torch.Tensor, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask))
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
        # RMSNorm / LayerNorm as defined in normalization.py
        dim = norm.w.shape[0]
        return {"dim": dim, "eps": norm.eps}
    # Fallback: try to read normalised_shape from nn.LayerNorm
    if hasattr(norm, "normalized_shape"):
        return {"dim": norm.normalized_shape[0],
                "eps": getattr(norm, "eps", 1e-5)}
    raise TypeError(
        f"Cannot extract constructor kwargs from norm type {type(norm).__name__}. "
        "Pass dim= and eps= explicitly or subclass to expose them."
    )
