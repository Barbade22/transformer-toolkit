# normalization.py
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Classic. Used in BERT, GPT-2."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, keepdim=True, unbiased=False)
        return self.w * (x - mean) / (var + self.eps).sqrt() + self.b


class RMSNorm(nn.Module):
    """No mean subtraction, no bias. Used in LLaMA, Mistral, Qwen."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.w * x / rms


class DeepNorm(nn.Module):
    """Scales residual before LayerNorm. Stabilizes very deep transformers (1000+ layers)."""
    def __init__(self, dim: int, alpha: float = 2.0):
        super().__init__()
        self.norm  = LayerNorm(dim)
        self.alpha = alpha

    def forward(self, x, residual):
        return self.norm(x + self.alpha * residual)