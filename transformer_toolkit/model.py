# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal

from .attention            import MultiHeadAttention, GroupedQueryAttention, \
                                  MultiQueryAttention, FlashAttention, MLAttention
from .feed_forward         import FFN, ReLUFFN, GLU, ReGLU, GeGLU, SwiGLU, MoE, \
                                  ExpertChoiceMoE, SharedExpertMoE
from .normalization        import LayerNorm, RMSNorm
from .positional_encodings import SinusoidalPE, LearnedPE, RoPE, ALiBi
from .block                import TransformerBlock


# ─── colors (inline so model.py has no dep on colors.py) ─────────────────────

class _C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    GREEN = "\033[32m"; CYAN = "\033[36m"; YELLOW = "\033[33m"
    BLUE  = "\033[34m"; RED  = "\033[31m"; WHITE  = "\033[37m"
    MAGENTA = "\033[35m"


# ─── config ───────────────────────────────────────────────────────────────────

@dataclass
class TransformerConfig:
    # core
    vocab_size: int   = 32000
    dim:        int   = 512
    n_layers:   int   = 8
    n_heads:    int   = 8
    max_seq:    int   = 2048

    # attention
    attn:       Literal["mha", "gqa", "mqa", "flash", "mla"] = "gqa"
    n_kv_heads: int   = 4        # gqa only
    latent_dim: int   = 64       # mla only

    # ffn
    ffn:        Literal["ffn", "relu_ffn", "glu", "reglu", "geglu",
                        "swiglu", "moe", "moe_ec", "moe_shared"] = "swiglu"
    hidden_dim: int   = None     # defaults to dim * 4
    n_experts:  int   = 8        # moe / moe_ec / moe_shared — total experts
    top_k:      int   = 2        # moe / moe_shared — experts per token
    moe_aux_weight:    float = 0.01  # moe / moe_shared load-balancing loss weight
    moe_capacity:      float = 1.0   # moe_ec capacity factor
    moe_n_shared:      int   = 2     # moe_shared — always-active experts
    moe_n_routed:      int   = 6     # moe_shared — sparse routed experts

    # normalization
    norm:       Literal["rmsnorm", "layernorm"] = "rmsnorm"
    eps:        float = 1e-6

    # positional encoding
    pos_enc:    Literal["rope", "sinusoidal", "learned", "alibi", "none"] = "rope"

    # regularisation
    dropout:    float = 0.0

    # output
    tie_weights: bool = True

    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = self.dim * 4

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


# ─── registry ─────────────────────────────────────────────────────────────────

def _build_norm(cfg: TransformerConfig) -> nn.Module:
    match cfg.norm:
        case "rmsnorm":   return RMSNorm(cfg.dim, cfg.eps)
        case "layernorm": return LayerNorm(cfg.dim, cfg.eps)
        case _:           raise ValueError(f"unknown norm: {cfg.norm!r}")


def _build_ffn(cfg: TransformerConfig) -> nn.Module:
    match cfg.ffn:
        case "ffn":        return FFN(cfg.dim, cfg.hidden_dim)
        case "relu_ffn":   return ReLUFFN(cfg.dim, cfg.hidden_dim)
        case "glu":        return GLU(cfg.dim, cfg.hidden_dim)
        case "reglu":      return ReGLU(cfg.dim, cfg.hidden_dim)
        case "geglu":      return GeGLU(cfg.dim, cfg.hidden_dim)
        case "swiglu":     return SwiGLU(cfg.dim, cfg.hidden_dim)
        case "moe":        return MoE(cfg.dim, cfg.hidden_dim, cfg.n_experts,
                                       cfg.top_k, cfg.moe_aux_weight)
        case "moe_ec":     return ExpertChoiceMoE(cfg.dim, cfg.hidden_dim,
                                                   cfg.n_experts, cfg.moe_capacity,
                                                   cfg.moe_aux_weight)
        case "moe_shared": return SharedExpertMoE(cfg.dim, cfg.hidden_dim,
                                                   cfg.moe_n_shared, cfg.moe_n_routed,
                                                   cfg.top_k, cfg.moe_aux_weight)
        case _:            raise ValueError(f"unknown ffn: {cfg.ffn!r}")


def _build_attn(cfg: TransformerConfig, pos_enc) -> nn.Module:
    rope = pos_enc if isinstance(pos_enc, RoPE) else None
    match cfg.attn:
        case "mha":   return MultiHeadAttention(cfg.dim, cfg.n_heads, rope)
        case "gqa":   return GroupedQueryAttention(cfg.dim, cfg.n_heads,
                                                    cfg.n_kv_heads, rope)
        case "mqa":   return MultiQueryAttention(cfg.dim, cfg.n_heads, rope)
        case "flash": return FlashAttention(cfg.dim, cfg.n_heads, rope)
        case "mla":   return MLAttention(cfg.dim, cfg.n_heads,
                                          cfg.latent_dim, rope)
        case _:       raise ValueError(f"unknown attn: {cfg.attn!r}")


def _build_pos_enc(cfg: TransformerConfig):
    match cfg.pos_enc:
        case "rope":       return RoPE(cfg.head_dim, cfg.max_seq)
        case "sinusoidal": return SinusoidalPE(cfg.dim, cfg.max_seq)
        case "learned":    return LearnedPE(cfg.dim, cfg.max_seq)
        case "alibi":      return ALiBi(cfg.n_heads)
        case "none":       return None
        case _:            raise ValueError(f"unknown pos_enc: {cfg.pos_enc!r}")


# ─── debug helpers ────────────────────────────────────────────────────────────

def _fmt(t: torch.Tensor) -> str:
    """One-line tensor summary: shape, dtype, min/mean/max, nan/inf check."""
    if t is None:
        return "None"
    f        = t.float()
    has_nan  = f.isnan().any().item()
    has_inf  = f.isinf().any().item()
    finite   = f[f.isfinite()]
    flags    = ""
    if has_nan: flags += f"  {_C.RED}NaN!{_C.RESET}"
    if has_inf: flags += f"  {_C.YELLOW}Inf!{_C.RESET}"
    if finite.numel() == 0:
        return (f"{_C.CYAN}{list(t.shape)}{_C.RESET}  {_C.DIM}{t.dtype}{_C.RESET}"
                f"  all non-finite{flags}")
    mn, mx, mean = finite.min().item(), finite.max().item(), finite.mean().item()
    return (
        f"{_C.CYAN}{list(t.shape)}{_C.RESET}  {_C.DIM}{t.dtype}{_C.RESET}"
        f"  min={_C.WHITE}{mn:+.3f}{_C.RESET}"
        f"  mean={_C.WHITE}{mean:+.3f}{_C.RESET}"
        f"  max={_C.WHITE}{mx:+.3f}{_C.RESET}"
        f"{flags}"
    )


def _grad_fmt(t: torch.Tensor) -> str:
    """One-line gradient summary."""
    if t is None or t.grad is None:
        return f"{_C.DIM}no grad{_C.RESET}"
    g = t.grad.float()
    norm = g.norm().item()
    has_nan = g.isnan().any().item()
    has_inf = g.isinf().any().item()
    flags   = ""
    if has_nan: flags += f"  {_C.RED}NaN grad!{_C.RESET}"
    if has_inf: flags += f"  {_C.YELLOW}Inf grad!{_C.RESET}"
    return (
        f"norm={_C.WHITE}{norm:.4f}{_C.RESET}"
        f"  mean={_C.WHITE}{g.mean().item():+.4f}{_C.RESET}"
        f"{flags}"
    )


# ─── model ────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig, debug: bool = False):
        super().__init__()
        self.cfg   = cfg
        self.debug = debug

        self.embed    = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.emb_drop = nn.Dropout(cfg.dropout)

        # RoPE stored as plain Python attribute — not a registered submodule.
        # Registering it would cause its buffers to appear twice in state_dict
        # (once here, once inside each attention block that holds a reference).
        # ALiBi / SinusoidalPE / LearnedPE registered normally — they have
        # parameters or persistent buffers that must survive save/load.
        _pe = _build_pos_enc(cfg)
        if isinstance(_pe, RoPE):
            self.__dict__['pos_enc'] = _pe
        else:
            self.pos_enc = _pe

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim     = cfg.dim,
                n_heads = cfg.n_heads,
                hidden  = cfg.hidden_dim,
                norm    = _build_norm(cfg),
                attn    = _build_attn(cfg, _pe),   # use _pe directly, not self.pos_enc
                ffn     = _build_ffn(cfg),
                dropout = cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])
        self.norm = _build_norm(cfg)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.head.weight = self.embed.weight

        if self.debug:
            self._print_model_summary()

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, aux_loss).
        aux_loss is non-zero only for MoE FFN — add to training loss:
            loss = ce_loss + aux_loss
        For non-MoE models aux_loss is 0.0 and has no effect.
        """
        D = self.debug
        w = 62

        if D:
            print(f"\n{_C.BOLD}{_C.MAGENTA}{'─'*w}{_C.RESET}")
            print(f"{_C.BOLD}{_C.MAGENTA}  🔬 Forward pass debug{_C.RESET}")
            print(f"{_C.BOLD}{_C.MAGENTA}{'─'*w}{_C.RESET}")
            print(f"  {_C.DIM}tokens  {_C.RESET} {_fmt(tokens)}")

        x = self.emb_drop(self.embed(tokens))

        if D:
            print(f"  {_C.DIM}embed   {_C.RESET} {_fmt(x)}")

        # stream encodings applied once to x
        if isinstance(self.pos_enc, (SinusoidalPE, LearnedPE)):
            x = self.pos_enc(x)
            if D:
                print(f"  {_C.DIM}pos_enc {_C.RESET} {_fmt(x)}  ({type(self.pos_enc).__name__})")

        # ALiBi: compute bias once and pass to every block
        mask = None
        if isinstance(self.pos_enc, ALiBi):
            mask = self.pos_enc.get_bias(tokens.shape[1], tokens.device)
            if D:
                print(f"  {_C.DIM}alibi   {_C.RESET} {_fmt(mask)}")

        if D and isinstance(self.pos_enc, RoPE):
            print(f"  {_C.DIM}pos_enc {_C.RESET} RoPE — applied inside attention to q,k")

        aux_loss = 0.0
        for i, block in enumerate(self.blocks):
            x_in = x
            x, block_aux = block(x, mask=mask)
            aux_loss = aux_loss + block_aux

            if D:
                # residual stream stats
                print(f"  {_C.DIM}block {i:<2}{_C.RESET}  in={_fmt(x_in)}  "
                      f"out={_fmt(x)}")
                # check residual update magnitude
                delta = (x - x_in).float().norm().item()
                x_norm = x_in.float().norm().item()
                ratio = delta / (x_norm + 1e-8)
                color = _C.GREEN if 0.01 < ratio < 2.0 else _C.RED
                print(f"           {_C.DIM}residual update norm ratio:{_C.RESET} "
                      f"{color}{ratio:.4f}{_C.RESET}"
                      + (f"  {_C.RED}← too small, vanishing?{_C.RESET}" if ratio < 0.01 else "")
                      + (f"  {_C.YELLOW}← large update{_C.RESET}" if ratio > 2.0 else ""))
                if isinstance(block_aux, torch.Tensor) and block_aux.item() > 0:
                    print(f"           {_C.DIM}moe aux_loss:{_C.RESET} "
                          f"{_C.CYAN}{block_aux.item():.4f}{_C.RESET}")

        normed = self.norm(x)
        logits = self.head(normed)

        if D:
            print(f"  {_C.DIM}norm    {_C.RESET} {_fmt(normed)}")
            print(f"  {_C.DIM}logits  {_C.RESET} {_fmt(logits)}")
            # logit entropy — high entropy = uniform = uncertain, low = peaked
            probs   = F.softmax(logits[:, -1, :].float(), dim=-1)
            entropy = -(probs * (probs + 1e-9).log()).sum(-1).mean().item()
            max_possible = torch.log(torch.tensor(self.cfg.vocab_size, dtype=torch.float)).item()
            print(f"  {_C.DIM}entropy {_C.RESET} {_C.WHITE}{entropy:.3f}{_C.RESET}"
                  f"  {_C.DIM}/ max {max_possible:.2f}{_C.RESET}"
                  f"  {_C.DIM}({100*entropy/max_possible:.1f}% of uniform){_C.RESET}")
            if isinstance(aux_loss, torch.Tensor):
                print(f"  {_C.DIM}aux_loss{_C.RESET} {_C.CYAN}{aux_loss.item():.4f}{_C.RESET}")
            print(f"{_C.BOLD}{_C.MAGENTA}{'─'*w}{_C.RESET}\n")

        return logits, aux_loss

    # ── generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        tokens:      torch.Tensor,
        max_new:     int   = 128,
        temperature: float = 1.0,
        top_k:       int   = 50,
    ) -> torch.Tensor:
        """
        Auto-regressive generation with top-k sampling.
        top_k=1  → greedy decoding.
        top_k=0  → full-vocab softmax (not recommended).
        """
        for _ in range(max_new):
            context     = tokens[:, -self.cfg.max_seq:]
            logits, _   = self.forward(context)
            logits      = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                k      = min(top_k, logits.size(-1))
                thresh = logits.topk(k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < thresh, float('-inf'))
            probs  = F.softmax(logits, dim=-1)
            tokens = torch.cat([tokens, torch.multinomial(probs, 1)], dim=-1)
        return tokens

    # ── utilities ─────────────────────────────────────────────────────────────

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # RoPE is not a registered submodule so .to() does not move it
        # automatically — move inv_freq to the new device and rebuild cache.
        # inv_freq is always kept as float32 regardless of model dtype —
        # frequency precision matters and bfloat16 has insufficient range.
        if isinstance(self.pos_enc, RoPE):
            device = next(self.parameters()).device
            self.pos_enc.inv_freq = self.pos_enc.inv_freq.to(
                device=device, dtype=torch.float32
            )
            self.pos_enc._build_cache(self.pos_enc.cos_cache.shape[2])
        return result

    def n_params(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{n/1e6:.2f}M" if n < 1e9 else f"{n/1e9:.2f}B"

    def debug_gradients(self):
        """
        Call after loss.backward() to inspect gradient health of every
        named parameter. Flags vanishing (<1e-6 norm) and exploding (>10) grads.
        Useful to run at step 1 and step 100 to confirm gradients are flowing.
        """
        w = 62
        print(f"\n{_C.BOLD}{_C.BLUE}{'─'*w}{_C.RESET}")
        print(f"{_C.BOLD}{_C.BLUE}  📐 Gradient debug{_C.RESET}")
        print(f"{_C.BOLD}{_C.BLUE}{'─'*w}{_C.RESET}")
        for name, param in self.named_parameters():
            if param.grad is None:
                print(f"  {_C.DIM}{name:<45}{_C.RESET} {_C.DIM}no grad{_C.RESET}")
                continue
            norm = param.grad.float().norm().item()
            if norm < 1e-6:
                flag = f"  {_C.RED}← vanishing!{_C.RESET}"
            elif norm > 10.0:
                flag = f"  {_C.YELLOW}← exploding?{_C.RESET}"
            else:
                flag = ""
            print(f"  {_C.DIM}{name:<45}{_C.RESET} {_grad_fmt(param)}{flag}")
        print(f"{_C.BOLD}{_C.BLUE}{'─'*w}{_C.RESET}\n")

    def debug_weights(self):
        """
        Print weight statistics for every parameter.
        Useful at init to confirm parameters are well-scaled,
        and after training to spot dead or saturated layers.
        """
        w = 62
        print(f"\n{_C.BOLD}{_C.CYAN}{'─'*w}{_C.RESET}")
        print(f"{_C.BOLD}{_C.CYAN}  ⚖️  Weight debug{_C.RESET}")
        print(f"{_C.BOLD}{_C.CYAN}{'─'*w}{_C.RESET}")
        for name, param in self.named_parameters():
            print(f"  {_C.DIM}{name:<45}{_C.RESET} {_fmt(param.data)}")
        print(f"{_C.BOLD}{_C.CYAN}{'─'*w}{_C.RESET}\n")

    def _print_model_summary(self):
        """Printed once at construction when debug=True."""
        cfg = self.cfg
        w   = 62
        print(f"\n{_C.BOLD}{_C.CYAN}{'─'*w}{_C.RESET}")
        print(f"{_C.BOLD}{_C.CYAN}  🏗️  Model summary{_C.RESET}")
        print(f"{_C.BOLD}{_C.CYAN}{'─'*w}{_C.RESET}")

        rows = [
            ("params",       self.n_params()),
            ("vocab_size",   f"{cfg.vocab_size:,}"),
            ("dim",          str(cfg.dim)),
            ("n_layers",     str(cfg.n_layers)),
            ("n_heads",      str(cfg.n_heads)),
            ("head_dim",     str(cfg.head_dim)),
            ("hidden_dim",   str(cfg.hidden_dim)),
            ("attn",         cfg.attn),
            ("ffn",          cfg.ffn),
            ("norm",         cfg.norm),
            ("pos_enc",      cfg.pos_enc),
            ("dropout",      str(cfg.dropout)),
            ("tie_weights",  str(cfg.tie_weights)),
            ("max_seq",      str(cfg.max_seq)),
        ]
        if cfg.attn == "gqa":
            rows.insert(6, ("n_kv_heads", str(cfg.n_kv_heads)))
        if cfg.attn == "mla":
            rows.insert(6, ("latent_dim", str(cfg.latent_dim)))
        if cfg.ffn == "moe":
            rows += [("n_experts", str(cfg.n_experts)),
                     ("top_k",     str(cfg.top_k)),
                     ("aux_weight",str(cfg.moe_aux_weight))]

        for k, v in rows:
            print(f"  {_C.DIM}{k:<18}{_C.RESET} {_C.WHITE}{v}{_C.RESET}")

        # parameter breakdown by component
        print(f"\n  {_C.DIM}parameter breakdown:{_C.RESET}")
        groups = {
            "embed":  self.embed,
            "blocks": self.blocks,
            "norm":   self.norm,
            "head":   self.head,
        }
        total = sum(p.numel() for p in self.parameters())
        for gname, mod in groups.items():
            n = sum(p.numel() for p in mod.parameters())
            # when tie_weights=True, head.weight is the same tensor as embed.weight
            # — PyTorch deduplicates in self.parameters() but not in mod.parameters(),
            # so subtract it here to avoid the bar chart summing over 100%
            if gname == "head" and cfg.tie_weights:
                n = 0
            bar_w = 20
            filled = int(bar_w * n / max(total, 1))
            bar = f"{_C.CYAN}{'█'*filled}{'░'*(bar_w-filled)}{_C.RESET}"
            pct = 100 * n / max(total, 1)
            print(f"  {_C.DIM}{gname:<8}{_C.RESET}  {bar}  "
                  f"{_C.WHITE}{n/1e6:.2f}M{_C.RESET}  {_C.DIM}{pct:.1f}%{_C.RESET}")

        print(f"{_C.BOLD}{_C.CYAN}{'─'*w}{_C.RESET}\n")

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def state_dict_for_save(self) -> dict:
        """
        Use instead of state_dict() when tie_weights=True.
        Removes head.weight so load re-applies the tie instead of
        breaking it into two independent tensors.
        """
        sd = self.state_dict()
        if self.cfg.tie_weights:
            sd.pop("head.weight", None)
        return sd

    def load_state_dict_with_tie(self, sd: dict, strict: bool = True):
        """Companion to state_dict_for_save(). Re-applies weight tying after load."""
        missing, unexpected = super().load_state_dict(sd, strict=False)
        expected_missing    = ["head.weight"] if self.cfg.tie_weights else []
        real_missing        = [k for k in missing if k not in expected_missing]
        if strict and (real_missing or unexpected):
            raise RuntimeError(
                f"Missing keys: {real_missing}  Unexpected keys: {unexpected}"
            )
        if self.cfg.tie_weights:
            self.head.weight = self.embed.weight