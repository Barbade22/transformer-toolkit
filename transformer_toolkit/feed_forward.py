# feed_forward.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── standard FFNs ────────────────────────────────────────────────────────────

class FFN(nn.Module):
    """
    Standard FFN with GELU. Used in original Transformer, BERT, GPT-2.
    dim → hidden → dim with GELU activation.
    """
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return self.net(x)


class ReLUFFN(nn.Module):
    """
    Standard FFN with ReLU. Used in original 'Attention is All You Need'.
    Kept separate from FFN for explicit architectural comparisons.
    """
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return self.net(x)


# ─── gated FFNs ───────────────────────────────────────────────────────────────

class GLU(nn.Module):
    """
    Gated Linear Unit. Base class for all gated FFN variants.
    gate(x) ⊙ value(x) — sigmoid gate controls information flow.
    Dauphin et al. 2017.
    """
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate
        self.w2 = nn.Linear(hidden, dim, bias=False)   # output
        self.w3 = nn.Linear(dim, hidden, bias=False)   # value

    def forward(self, x):
        return self.w2(torch.sigmoid(self.w1(x)) * self.w3(x))


class ReGLU(nn.Module):
    """
    Gated FFN with ReLU gate. ReLU(gate) ⊙ value.
    Noam Shazeer 2020 — 'GLU Variants Improve Transformer'.
    """
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)) * self.w3(x))


class GeGLU(nn.Module):
    """
    Gated FFN with GELU gate. GELU(gate) ⊙ value.
    Used in PaLM, T5v1.1, Flan-T5.
    Slightly smoother than SwiGLU — GELU vs SiLU as activation.
    Noam Shazeer 2020.
    """
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)) * self.w3(x))


class SwiGLU(nn.Module):
    """
    Gated FFN with SiLU (Swish) gate. SiLU(gate) ⊙ value.
    Used in LLaMA 1/2/3, Mistral, PaLM, Gemma, Qwen.
    Currently the most widely used FFN in production LLMs.
    Noam Shazeer 2020.
    """
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ─── mixture of experts ───────────────────────────────────────────────────────

class MoE(nn.Module):
    """
    Token-Choice Mixture of Experts (Sparse MoE).
    Each token routes itself to the top-k experts.
    Used in Mixtral 8x7B, GPT-4 (rumoured).

    Returns (output, aux_loss). aux_loss is already scaled by aux_weight.
    Add it directly to your training loss:
        logits, aux = model(x)
        loss = F.cross_entropy(...) + aux

    Without aux_loss, routing collapses to 1-2 experts within ~200 steps.
    aux_weight=0.01 is a safe default (Mixtral uses 0.02).
    """
    def __init__(self, dim: int, hidden: int, n_experts: int, top_k: int = 2,
                 aux_weight: float = 0.01):
        super().__init__()
        self.top_k      = top_k
        self.n_experts  = n_experts
        self.aux_weight = aux_weight
        self.gate       = nn.Linear(dim, n_experts, bias=False)
        self.experts    = nn.ModuleList([SwiGLU(dim, hidden) for _ in range(n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)                               # [N, C]

        gate_logits  = self.gate(x_flat)                     # [N, n_experts]
        weights, idx = gate_logits.topk(self.top_k, dim=-1)  # [N, top_k]
        weights      = F.softmax(weights, dim=-1)             # [N, top_k]

        out = torch.zeros_like(x_flat)

        # n_experts kernel launches (not n_experts × top_k)
        for i, expert in enumerate(self.experts):
            expert_mask = (idx == i)                          # [N, top_k]
            token_mask  = expert_mask.any(dim=-1)             # [N]
            if not token_mask.any():
                continue
            w = (weights * expert_mask.float()).sum(dim=-1, keepdim=True)
            out[token_mask] += w[token_mask] * expert(x_flat[token_mask])

        # Switch Transformer load-balancing loss
        router_probs = F.softmax(gate_logits, dim=-1)         # [N, n_experts]
        ones         = torch.zeros_like(router_probs)
        ones.scatter_(1, idx[:, 0:1], 1.0)
        f_i      = ones.mean(dim=0)
        p_i      = router_probs.mean(dim=0)
        aux_loss = self.aux_weight * self.n_experts * (f_i * p_i).sum()

        return out.view(B, T, C), aux_loss


class ExpertChoiceMoE(nn.Module):
    """
    Expert-Choice MoE — experts pick tokens instead of tokens picking experts.
    Used in Google's Expert Choice paper (2022).

    Key difference from token-choice MoE:
    - Each expert selects its top-k tokens from the batch
    - Guarantees perfect load balancing by design (no aux loss needed)
    - Each token may be processed by 0, 1, or multiple experts
    - Better utilization than token-choice on long sequences

    capacity_factor controls how many tokens each expert processes:
        tokens_per_expert = int(capacity_factor * N / n_experts)
    capacity_factor=1.0 means each token is processed by exactly 1 expert on average.

    Returns (output, aux_loss=0.0) — no aux loss needed, balanced by construction.
    """
    def __init__(self, dim: int, hidden: int, n_experts: int,
                 capacity_factor: float = 1.0, aux_weight: float = 0.0):
        super().__init__()
        self.n_experts       = n_experts
        self.capacity_factor = capacity_factor
        self.aux_weight      = aux_weight   # kept for API consistency, not used
        self.gate            = nn.Linear(dim, n_experts, bias=False)
        self.experts         = nn.ModuleList([SwiGLU(dim, hidden) for _ in range(n_experts)])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)                                # [N, C]
        N      = x_flat.shape[0]

        # capacity = how many tokens each expert processes
        capacity = max(1, int(self.capacity_factor * N / self.n_experts))

        gate_logits = self.gate(x_flat)                       # [N, n_experts]
        # transpose: each expert sees scores for all tokens [n_experts, N]
        scores      = gate_logits.T                           # [n_experts, N]

        # each expert picks its top-capacity tokens
        _, token_idx = scores.topk(capacity, dim=-1)          # [n_experts, capacity]
        # softmax weights for selected tokens
        token_weights = F.softmax(
            scores.gather(1, token_idx), dim=-1
        )                                                     # [n_experts, capacity]

        out = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            idx_i = token_idx[i]                             # [capacity]
            w_i   = token_weights[i].unsqueeze(-1)           # [capacity, 1]
            out.index_add_(0, idx_i,
                           w_i * expert(x_flat[idx_i]))

        # no aux loss — load balancing guaranteed by construction
        return out.view(B, T, C), torch.tensor(0.0, device=x.device)


class SharedExpertMoE(nn.Module):
    """
    Shared Expert MoE — DeepSeek-V2/V3 architecture.

    Combines:
    - n_shared shared experts that ALWAYS run for every token
    - n_routed sparse experts where each token routes to top_k

    shared experts  → learn general, universal patterns
    routed experts  → learn specialised, token-specific patterns

    Advantages over pure sparse MoE:
    - Shared experts prevent routing collapse (they always train)
    - Better knowledge sharing across all tokens
    - Sparse experts can truly specialise without worrying about basics

    Returns (output, aux_loss) — aux_loss from routed experts only.
    """
    def __init__(self, dim: int, hidden: int,
                 n_shared: int = 2,
                 n_routed: int = 6,
                 top_k: int = 2,
                 aux_weight: float = 0.01):
        super().__init__()
        self.top_k      = top_k
        self.n_routed   = n_routed
        self.aux_weight = aux_weight

        # shared experts — always active, no gating
        self.shared_experts = nn.ModuleList(
            [SwiGLU(dim, hidden) for _ in range(n_shared)]
        )

        # routed experts — sparse, token-choice top-k
        self.gate           = nn.Linear(dim, n_routed, bias=False)
        self.routed_experts = nn.ModuleList(
            [SwiGLU(dim, hidden) for _ in range(n_routed)]
        )

        # output projection merges shared + routed contributions
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)                                # [N, C]

        # ── shared path — every token goes through all shared experts ──
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out += expert(x_flat)

        # ── routed path — top-k sparse routing ──
        gate_logits  = self.gate(x_flat)                      # [N, n_routed]
        weights, idx = gate_logits.topk(self.top_k, dim=-1)
        weights      = F.softmax(weights, dim=-1)

        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            expert_mask = (idx == i)
            token_mask  = expert_mask.any(dim=-1)
            if not token_mask.any():
                continue
            w = (weights * expert_mask.float()).sum(dim=-1, keepdim=True)
            routed_out[token_mask] += w[token_mask] * expert(x_flat[token_mask])

        # combine shared + routed
        combined = shared_out + routed_out

        # ── load-balancing aux loss on routed experts only ──
        router_probs = F.softmax(gate_logits, dim=-1)
        ones         = torch.zeros_like(router_probs)
        ones.scatter_(1, idx[:, 0:1], 1.0)
        f_i      = ones.mean(dim=0)
        p_i      = router_probs.mean(dim=0)
        aux_loss = self.aux_weight * self.n_routed * (f_i * p_i).sum()

        return self.out(combined).view(B, T, C), aux_loss