# Transformer-Toolkit

[▶ Watch Video](https://github.com/user-attachments/assets/96f4576b-6c90-4d31-b0cb-c528c8bb8b3b)

> **Music Credit**: Background music "Let's Go" by Elysium Audio Labs. See [MUSIC_CREDITS.md](MUSIC_CREDITS.md) for details.

<p align="center">
  <a href="https://pypi.org/project/transformer-toolkit/"><img src="https://img.shields.io/pypi/v/transformer-toolkit?color=cyan&style=flat-square" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/transformer-toolkit/"><img src="https://static.pepy.tech/badge/transformer-toolkit" alt="Downloads"/></a>
  <a href="https://github.com/Barbade22/transformer-toolkit/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Barbade22/transformer-toolkit?style=flat-square" alt="License"/></a>
</p>

## Overview

**Transformer-Toolkit** is a production-ready, modular transformer library from scratch for training and experimenting with modern LLM architectures. Build compact, customizable models with swappable components — attention mechanisms, positional encodings, feed-forward networks, and normalization strategies — all configured via a single `TransformerConfig` object.

### Why Transformer-Toolkit?

- **Fully modular**: Pick your attention type, FFN variant, positional encoding, and normalization independently
- **Production features**: Mixed precision training, gradient checkpointing, KV caching, HuggingFace Hub integration
- **SFT support**: Full supervised fine-tuning pipeline with multi-turn conversation handling and loss masking
- **Fast tokenizers**: Rust-backed BPE tokenizer trained 100x faster than pure Python
- **Efficient dataloading**: Memmap-based dataset loading supports GB-scale datasets with zero RAM overhead
- **Clear inference API**: Temperature, top-k, top-p, repetition penalty controls out-of-the-box

```bash
pip install transformer-toolkit
```

---

## Table of Contents

- [Core Modules](#core-modules)
  - [Transformer Model](#transformer-model)
  - [Attention Mechanisms](#attention-mechanisms)
  - [Feed-Forward Networks](#feed-forward-networks)
  - [Normalization Layers](#normalization-layers)
  - [Positional Encodings](#positional-encodings)
  - [Transformer Block](#transformer-block)
- [Data & Tokenization](#data--tokenization)
  - [Tokenizers](#tokenizers)
  - [Dataloader](#dataloader)
- [Training](#training)
  - [Pretraining](#pretraining)
  - [Supervised Fine-Tuning](#supervised-fine-tuning-sft)
  - [Chat Templates](#chat-templates)
- [Inference & Utilities](#inference--utilities)
  - [Inference Engine](#inference-engine)
  - [HuggingFace Hub Integration](#huggingface-hub-integration)
  - [Color Utilities](#color-utilities)
- [Examples](#examples)
- [API Reference](#api-reference)

---

## Core Modules

### Transformer Model

**Location**: `transformer_toolkit/model.py`  
**Main Classes**: `Transformer`, `TransformerConfig`, `TransformerBlock`

The core transformer model built from composable modules. A single `TransformerConfig` object controls every architectural decision.

#### TransformerConfig

Complete model configuration via a dataclass. All attributes are optional with sensible defaults.

```python
from transformer_toolkit.model import TransformerConfig

cfg = TransformerConfig(
    # ── core ──────────────────────────────────────────────────────────
    vocab_size  = 32000,      # tokenizer vocabulary size
    dim         = 512,        # model embedding dimension
    n_layers    = 8,          # number of transformer blocks
    n_heads     = 8,          # number of attention heads
    max_seq     = 2048,       # maximum sequence length

    # ── attention ─────────────────────────────────────────────────────
    attn       = "gqa",      # "mha" | "gqa" | "mqa" | "flash" | "mla"
    n_kv_heads = 4,          # gqa only — n_heads must be divisible by n_kv_heads
    latent_dim = 64,         # mla only — latent compression dimension

    # ── feed-forward ──────────────────────────────────────────────────
    ffn        = "swiglu",   # "ffn" | "relu_ffn" | "glu" | "reglu" | "geglu"
                             # | "swiglu" | "moe" | "moe_ec" | "moe_shared"
    hidden_dim = 2048,       # FFN inner dimension (default: dim × 4)
    n_experts  = 8,          # moe / moe_ec / moe_shared — total experts
    top_k      = 2,          # moe / moe_shared — experts activated per token
    moe_aux_weight = 0.01,   # moe / moe_shared — load-balancing loss coefficient
    moe_capacity   = 1.0,    # moe_ec — capacity factor
    moe_n_shared   = 2,      # moe_shared — always-active experts
    moe_n_routed   = 6,      # moe_shared — sparse routed experts

    # ── normalization ─────────────────────────────────────────────────
    norm       = "rmsnorm",  # "rmsnorm" | "layernorm"
    eps        = 1e-6,

    # ── positional encoding ───────────────────────────────────────────
    pos_enc    = "rope",     # "rope" | "sinusoidal" | "learned" | "alibi" | "none"

    # ── regularisation ────────────────────────────────────────────────
    dropout    = 0.0,        # 0.0 recommended for SFT and inference

    # ── output ────────────────────────────────────────────────────────
    tie_weights = True,      # share embedding and output projection weights

    # ── inference ─────────────────────────────────────────────────────
    use_kv_cache = False,    # enable KV cache during generation (inference only)
)
```

**Key attributes:**

| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `vocab_size` | int | 32000 | Tokenizer vocabulary size |
| `dim` | int | 512 | Model hidden dimension |
| `n_layers` | int | 8 | Number of transformer blocks |
| `n_heads` | int | 8 | Number of attention heads |
| `max_seq` | int | 2048 | Maximum sequence length |
| `attn` | str | "gqa" | Attention mechanism variant |
| `ffn` | str | "swiglu" | Feed-forward network variant |
| `norm` | str | "rmsnorm" | Normalization layer type |
| `pos_enc` | str | "rope" | Positional encoding type |
| `tie_weights` | bool | True | Share embedding and output weights |

#### Transformer

Main model class. Initialize with a config and send to device.

```python
import torch
from transformer_toolkit.model import Transformer, TransformerConfig

cfg = TransformerConfig(vocab_size=8000, dim=384, n_layers=6)
model = Transformer(cfg).to("cuda")

# Get parameter count (human readable)
print(model.n_params())  # "12.45M"

# Forward pass — returns (logits, aux_loss)
logits, aux_loss = model(tokens)   # tokens: [B, T]  →  logits: [B, T, vocab_size]

# Generation with temperature, top-k, top-p
output = model.generate(
    tokens      = prompt_ids,      # [B, T]
    max_new     = 200,
    temperature = 0.8,
    top_k       = 40,
)
```

**Methods:**

- `forward(tokens)` → `(logits, aux_loss)`: Main forward pass. Returns logits and auxiliary MoE loss (0.0 if not MoE).
- `generate(tokens, max_new, temperature, top_k, top_p)` → `tokens`: Auto-regressive generation with sampling.
- `n_params()` → `str`: Human-readable parameter count.
- `debug_gradients()` → `None`: Print gradient statistics per parameter (call after `loss.backward()`).
- `debug_weights()` → `None`: Print weight statistics per parameter.
- `state_dict_for_save()` → `dict`: For weight-tied models, strips redundant weights before saving.
- `load_state_dict_with_tie()` → `None`: For weight-tied models, restores weights correctly after loading.

#### Weight Tying

Weight tying shares the embedding matrix with the output projection, reducing parameters. **However, it requires careful initialization**.

When tied, `nn.Embedding` initializes with `N(0, 1)` — values around ±5. Without scaling, this produces logits of ±400 instead of ±3, crashing training.

**Recommended**: Disable tying for training from scratch:

```python
cfg = TransformerConfig(tie_weights=False)
```

**If enabling tying**, scale the embedding at init:

```python
model = Transformer(cfg).to("cuda")
if cfg.tie_weights:
    with torch.no_grad():
        model.embed.weight.mul_(0.02)  # scale into ±3 range
```

#### Debug Mode

Enable `debug=True` to inspect model structure and forward pass:

```python
model = Transformer(cfg, debug=True).to("cuda")
# Prints model summary at construction:
#  🏗️  Model summary
#  params             16.35M
#  dim                384
#  n_layers           6
#  entropy check → should be > 90% of log(vocab_size) at init
```

Turn off after inspecting (runs on every forward pass):

```python
model.debug = False
```

---

### Attention Mechanisms

**Location**: `transformer_toolkit/attention.py`

Five attention variants, all swappable via `TransformerConfig.attn`. Pick the right one for your use case:

| Value | Class | Key property | Used in |
|-------|-------|--------------|---------|
| `"mha"` | `MultiHeadAttention` | Full KV cache per head | Original Transformer, BERT, GPT-2 |
| `"gqa"` | `GroupedQueryAttention` | Grouped KV heads (4x-8x faster) | LLaMA 3, Mistral |
| `"mqa"` | `MultiQueryAttention` | Single KV head (fastest) | Falcon, early Gemini |
| `"flash"` | `FlashAttention` | Fused CUDA kernels | All (PyTorch ≥ 2.0) |
| `"mla"` | `MLAttention` | Latent compression | DeepSeek-V2/V3 |

#### Multi-Head Attention (MHA)

Classic attention from the original Transformer. Each head has separate K/V caches.

```python
cfg = TransformerConfig(
    dim     = 512,
    n_heads = 8,
    attn    = "mha",  # full cache: [B, n_heads, T, head_dim]
)
```

**Good for**: Small models where memory is not a constraint.

#### Grouped Query Attention (GQA)

Multiple query heads share fewer key-value heads. Reduces KV cache size by `n_heads / n_kv_heads`.

```python
cfg = TransformerConfig(
    dim        = 512,
    n_heads    = 8,
    attn       = "gqa",
    n_kv_heads = 2,   # 4 query groups, each sharing 1 KV head
)
# Constraint: n_heads % n_kv_heads == 0
```

**Good for**: Production models. LLaMA-3 uses `n_heads=128, n_kv_heads=8` for massive KV cache reduction.

#### Multi-Query Attention (MQA)

All heads share a single K/V head. Fastest for inference, less expressive.

```python
cfg = TransformerConfig(
    dim        = 512,
    n_heads    = 8,
    attn       = "mqa",
    n_kv_heads = 1,   # all 8 heads share 1 KV head
)
```

#### Flash Attention

Uses PyTorch's fused `scaled_dot_product_attention` (requires PyTorch ≥ 2.0 + CUDA). Faster and more memory efficient.

```python
cfg = TransformerConfig(
    dim     = 512,
    n_heads = 8,
    attn    = "flash",  # uses fused kernel — no extra KV head config
)
```

#### Multi-Latent Attention (MLA)

DeepSeek's variant. Compresses Q/K/V to a lower-dimensional latent space via `latent_dim`.

```python
cfg = TransformerConfig(
    dim        = 512,
    n_heads    = 8,
    attn       = "mla",
    latent_dim = 128,   # compression dimension — typically dim/4
)
```

**RoPE (Rotary Position Encoding)** is applied inside each attention module to Q and K after head-splitting. Applied once per forward pass and shared across all layers.

**ALiBi bias** is computed once per forward pass and passed as an additive mask to every attention block.

**Causal masking** is applied automatically. No manual mask needed for standard language model training.

---

### Feed-Forward Networks

**Location**: `transformer_toolkit/feed_forward.py`

Nine FFN variants, each trading off between simplicity and expressiveness.

| Class | Activation | Formula | Use case |
|-------|------------|---------|----------|
| `FFN` | GELU | dense → GELU → dense | Original Transformer |
| `ReLUFFN` | ReLU | dense → ReLU → dense | Classic (older) |
| `GLU` | Sigmoid | (dense ⊙ sigmoid(dense)) | Simple gating |
| `ReGLU` | ReLU gating | (dense ⊗ ReLU(dense)) | ReLU-gated variant |
| `GeGLU` | GELU gating | (dense ⊗ GELU(dense)) | GELU-gated variant |
| `SwiGLU` | Swish gating | (dense ⊗ Swish(dense)) | LLaMA, Mistral, Qwen (recommended) |
| `MoE` | Sparse routing | k-of-n experts (load balanced) | Sparse scaling |
| `MoE_EC` | Expert choice | Token-to-expert assignment | Balanced capacity |
| `MoE_Shared` | Hybrid | Always-active + sparse experts | Best of both |

#### Standard FFNs

```python
# Original Transformer
cfg = TransformerConfig(ffn="ffn", hidden_dim=2048)

# ReLU variant (older)
cfg = TransformerConfig(ffn="relu_ffn", hidden_dim=2048)
```

#### Gated FFNs

Gated FFNs learn to selectively activate subsets of parameters. Generally outperform standard FFN.

```python
# SwiGLU — most popular (LLaMA, Mistral, Qwen)
cfg = TransformerConfig(ffn="swiglu", hidden_dim=2048)

# Other gates
cfg = TransformerConfig(ffn="geglu", hidden_dim=2048)   # GELU gate
cfg = TransformerConfig(ffn="reglu", hidden_dim=2048)   # ReLU gate
cfg = TransformerConfig(ffn="glu", hidden_dim=2048)     # Sigmoid gate
```

#### Mixture of Experts (MoE)

Conditionally activates only `top_k` out of `n_experts` experts per token. Dramatically scales parameter count without scaling compute.

**Standard MoE** — Each token independently chooses its top-k experts. Requires load-balancing loss to prevent expert collapse.

```python
cfg = TransformerConfig(
    ffn            = "moe",
    n_experts      = 8,
    top_k          = 2,
    moe_aux_weight = 0.01,   # Mixtral uses 0.02
)

logits, aux_loss = model(tokens)
ce_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
loss = ce_loss + aux_loss   # Always add aux_loss for MoE
```

The `Trainer` handles `aux_loss` automatically — no changes needed.

**Expert Choice MoE (moe_ec)** — Experts choose which tokens they process, not vice versa. Better load balancing and lower variance.

```python
cfg = TransformerConfig(
    ffn           = "moe_ec",
    n_experts     = 8,
    moe_capacity  = 1.25,    # capacity factor
)
```

**Shared Expert MoE (moe_shared)** — Hybrid approach. Some experts are always active, some are sparse.

```python
cfg = TransformerConfig(
    ffn         = "moe_shared",
    n_experts   = 8,          # total experts
    n_shared    = 2,          # always active
    n_routed    = 6,          # sparse routed
    top_k       = 2,          # routed tokens choose top-k
)
```

---

### Normalization Layers

**Location**: `transformer_toolkit/normalization.py`

Two normalization options, each with tradeoffs:

| Class | Subtraction | Bias | Scaling | Speed | Used in |
|-------|-------------|------|---------|-------|---------|
| `LayerNorm` | Yes (μ) | Yes | Yes | Slower | BERT, GPT-2 |
| `RMSNorm` | No | No | Yes | Faster | LLaMA, Mistral, Qwen |
| `DeepNorm` | Both per block | Both | Both | Slower | 1000+ layer transformers |

#### LayerNorm

Classic normalization. Subtracts mean and divides by standard deviation.

```python
from transformer_toolkit.normalization import LayerNorm

norm = LayerNorm(dim=512, eps=1e-5)
x_norm = norm(x)
```

Formula: $\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$

**Good for**: Training stability, better for smaller models.

#### RMSNorm

Root Mean Square normalization. No mean subtraction, no bias — faster and cleaner.

```python
from transformer_toolkit.normalization import RMSNorm

norm = RMSNorm(dim=512, eps=1e-6)
x_norm = norm(x)
```

Formula: $\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\text{RMS}(x)^2 + \epsilon}}$

**Good for**: Modern LLMs (LLaMA, Mistral, Qwen). Slightly lower memory, same training stability.

#### DeepNorm

Specialized for very deep transformers (1000+ layers). Scales residuals before normalization.

```python
from transformer_toolkit.normalization import DeepNorm

norm = DeepNorm(dim=512, alpha=2.0)
x_norm = norm(x, residual)
```

**Good for**: Ultra-deep models. Not needed for typical 6-32 layer models.

---

### Positional Encodings

**Location**: `transformer_toolkit/positional_encodings.py`

Five positional encoding strategies. RoPE is applied inside attention; others are applied to the residual stream before the first block.

| Type | Location | Learnable | Best for |
|------|----------|-----------|----------|
| `RoPE` | Inside attention (q, k) | No | Modern models (LLaMA, Mistral, Qwen) |
| `SinusoidalPE` | Residual stream | No | Original Transformer |
| `LearnedPE` | Residual stream | Yes | BERT, GPT-2 |
| `ALiBi` | Attention scores | No | Length generalization |
| `none` | Not applied | — | Ablation studies |

#### RoPE (Rotary Position Encoding)

Applies rotation matrices to query and key vectors inside attention. No learnable parameters.

```python
cfg = TransformerConfig(pos_enc="rope")  # default
```

**Advantages**: Enables length extrapolation (infer beyond training length), used in LLaMA, Mistral, Qwen.

**Details**: Applied after head-splitting, so each attention head gets independent rotations.

#### Sinusoidal Positional Encoding

Fixed sine/cosine patterns added to embeddings before the transformer blocks. From the original Transformer paper.

```python
cfg = TransformerConfig(pos_enc="sinusoidal", max_seq=2048)
```

Formula: $PE_{(pos, 2i)} = \sin(\text{pos} / 10000^{2i/d})$, $PE_{(pos, 2i+1)} = \cos(\text{pos} / 10000^{2i/d})$

#### Learned Positional Encoding

Trainable embedding table for positions. Used in BERT and GPT-2.

```python
cfg = TransformerConfig(pos_enc="learned", max_seq=2048)
```

#### ALiBi (Attention with Linear Biases)

Adds linear biases to attention scores based on relative position. No learnable parameters, supports arbitrary lengths.

```python
cfg = TransformerConfig(pos_enc="alibi")
```

**Advantages**: Enables length generalization without training on longer sequences.

#### No Positional Encoding

Ablation option — model receives no position information.

```python
cfg = TransformerConfig(pos_enc="none")  # for ablation studies
```

---

### Transformer Block

**Location**: `transformer_toolkit/block.py`

A single transformer block. Pre-norm architecture: `norm → attention → residual → norm → ffn → residual`.

**Key features**:
- Gradient checkpointing — trade compute for memory (~20% slower, 60% less VRAM)
- KV caching — for fast inference
- Auxiliary MoE loss — if using MoE FFN
- Flexible component swapping — inject any attention, FFN, norm

```python
from transformer_toolkit.block import TransformerBlock

block = TransformerBlock(
    dim            = 512,
    n_heads        = 8,
    hidden         = 2048,
    norm           = None,      # None = default to LayerNorm
    attn           = None,      # None = default to MultiHeadAttention
    ffn            = None,      # None = default to FFN
    dropout        = 0.1,
    use_checkpoint = False,     # enable for large models
)

# Forward returns (output, aux_loss, present_kv)
x, aux_loss, present_kv = block(x, past_kv=None)
```

**Gradient checkpointing** (for memory efficiency):

```python
block = TransformerBlock(
    ..., use_checkpoint=True
)
# Recomputes activations during backward — saves ~60% VRAM, ~20% slower
```

---

## Data & Tokenization

### Tokenizers

**Location**: `transformer_toolkit/c_tokenizers.py`

Three tokenizer classes with a unified interface. Each implements `encode()`, `decode()`, `train()`, `save()`, `load()`.

```python
from transformer_toolkit.c_tokenizers import (
    ByteLevelTokenizer,
    RustBPETokenizer,
    HFTokenizer,
)
```

#### ByteLevelTokenizer

Zero dependencies. Every byte (0-255) is a token. Works on any text or encoding immediately.

```python
tok = ByteLevelTokenizer()
ids = tok.encode("Hello")           # [72, 101, 108, 108, 111]
txt = tok.decode(ids)               # "Hello"
print(tok.vocab_size)               # 256
```

**Pros**: Universal, zero setup.  
**Cons**: Inefficient — long sequences need many tokens.

#### RustBPETokenizer

Byte-Pair Encoding backed by HuggingFace's Rust tokenizers library. **~100x faster than pure Python BPE**.

**Installation:**
```bash
pip install tokenizers
```

**Usage:**
```python
from transformer_toolkit.c_tokenizers import RustBPETokenizer

tok = RustBPETokenizer()

# Train once
tok.train(
    texts=open("data.txt", encoding="utf-8").readlines(),
    vocab_size=8000
)
tok.save("tokenizer.json")

# On subsequent runs — just load
tok = RustBPETokenizer()
tok.load("tokenizer.json")

ids = tok.encode("Hello world")
txt = tok.decode(ids)
print(tok.vocab_size)  # 8000
```

**Special tokens**: All SFT-related special tokens (chat format tokens, BOS, EOS, PAD) are registered automatically at train time.

```python
tok.train(texts=lines, vocab_size=32000)
# Special tokens registered automatically at fixed IDs:
# ID 0: [UNK], ID 1: [PAD], ID 2: [BOS], ID 3: [EOS]
# ID 7: <|im_start|>, ID 8: <|im_end|>  (ChatML)
# ID 9: <|start_header_id|>, ID 10: <|end_header_id|>  (LLaMA3)
# ... and more (see README "Chat Templates" section)
```

**Call `tok.validate_template()` before SFT to ensure all special tokens are properly registered:**

```python
from transformer_toolkit.chat_template import ChatTemplate

template = ChatTemplate("llama3")
tok.validate_template(template)
# Raises if special tokens are fragmented (not single vocab entries)
```

#### HFTokenizer

Thin wrapper around any HuggingFace pretrained tokenizer. Access thousands of pretrained tokenizers.

**Installation:**
```bash
pip install transformers
```

**Usage:**
```python
from transformer_toolkit.c_tokenizers import HFTokenizer

tok = HFTokenizer("gpt2")

ids = tok.encode("Hello world")
txt = tok.decode(ids)
print(tok.vocab_size)  # 50257

# Load any HuggingFace tokenizer
tok = HFTokenizer("meta-llama/Llama-2-7b-hf")
tok = HFTokenizer("mistralai/Mistral-7B-Instruct-v0.1")
```

---

### Dataloader

**Location**: `transformer_toolkit/dataloader.py`

Efficient data pipeline for training. Supports multiple sources and loading strategies, with memmap for minimal memory overhead on large datasets.

#### DataConfig

```python
from transformer_toolkit.dataloader import DataConfig

cfg = DataConfig(
    seq_len     = 512,        # sequence length fed to model
    batch_size  = 32,         # samples per batch
    split       = 0.9,        # fraction for training (rest for validation)
    stride      = None,       # None = non-overlapping; int = overlapping windows
    shuffle     = True,       # shuffle training data
    num_workers = 4,          # parallel dataloading workers
    pin_memory  = True,       # pin tensors to GPU memory
    debug       = False,      # print sample preview
    debug_n     = 3,          # number of debug samples
)
```

**Key attributes:**

| Attribute | Default | Purpose |
|-----------|---------|---------|
| `seq_len` | 512 | Sequence length for model |
| `batch_size` | 32 | Samples per batch |
| `split` | 0.9 | Train/val split ratio |
| `stride` | None | Window stride (None = seq_len, no overlap) |
| `shuffle` | True | Shuffle training batches |
| `num_workers` | 4 | Parallel loading workers |
| `debug` | False | Print decoded sample preview |

**stride parameter:**

- `stride=None` (default): Non-overlapping windows, few clean samples
- `stride=<int>`: Overlapping windows, many samples but faster overfitting on small data

**Example**: 1.86M tokens with `seq_len=128`:
- `stride=None` (128): ~14,600 samples
- `stride=1`: ~1.86M samples (rapid overfitting)

#### Loading from a Binary File

**One-time tokenization:**

```python
from transformer_toolkit.dataloader import save_binary, from_binary

# Tokenize once, save binary
tokens = tok.encode(open("data.txt", encoding="utf-8").read())
save_binary(tokens, "data.bin")

# Load anytime
cfg = DataConfig(seq_len=128, batch_size=32)
train_dl, val_dl = from_binary("data.bin", cfg, tokenizer=tok)
```

**With automatic NPY split (recommended for reuse):**

```python
train_dl, val_dl = from_binary(
    "data.bin", cfg,
    train_path="train.npy",  # saves splits here
    val_path="val.npy",
    tokenizer=tok,
)
```

On subsequent runs, load the pre-split `.npy` files directly (zero latency).

#### Memmap — Load Pre-split NPY Files

After first run, skip tokenization. The `.npy` files stay on disk — only accessed pages load into RAM. **Scales to 100GB+ datasets.**

```python
from transformer_toolkit.dataloader import from_npy_split

cfg = DataConfig(seq_len=512, batch_size=32)
train_dl, val_dl = from_npy_split(
    "train.npy", "val.npy", cfg, tokenizer=tok
)
```

#### Loading from Text Files

Multiple text files:

```python
from transformer_toolkit.dataloader import from_files

train_dl, val_dl = from_files(
    paths=["data1.txt", "data2.txt", "data3.txt"],
    tokenizer=tok,
    cfg=cfg,
    train_path="train.npy",  # optional — saves splits for future reuse
    val_path="val.npy",
    bos_id=tok.bos_id,       # optional — wrap documents with BOS/EOS
    eos_id=tok.eos_id,
)
```

#### Loading from HuggingFace

**Streaming** (no disk required, works with infinite datasets):

```python
from transformer_toolkit.dataloader import from_hf

cfg_stream = DataConfig(seq_len=512, batch_size=16, streaming=True)
train_dl, val_dl = from_hf(
    dataset_name="roneneldan/TinyStories",
    tokenizer=tok,
    cfg=cfg_stream,
)
```

**In-memory** (download fully, split, optionally save as `.npy`):

```python
train_dl, val_dl = from_hf(
    dataset_name="roneneldan/TinyStories",
    tokenizer=tok,
    cfg=cfg,
    text_col="text",           # column containing text
    bos_id=tok.bos_id,
    eos_id=tok.eos_id,
    train_path="train.npy",    # save splits for future memmap loads
    val_path="val.npy",
)
```

#### Dataloader Debug Mode

Preview decoded samples before training:

```python
cfg = DataConfig(seq_len=128, batch_size=32, debug=True, debug_n=2)
train_dl, val_dl = from_binary("data.bin", cfg, tokenizer=tok)
```

**Output:**
```
🔍 Debug samples (train)
seq_len=128  stride=128  batch_size=32

sample 1
x ids : [23, 451, 12, 8, 1203 ...] ... +121
y ids : [451, 12, 8, 1203, 44 ...] ... +121
x text: 'ROMEO:\nBut soft, what light through yonder window...'
y text: '\nBut soft, what light through yonder window breaks'
✓  x/y alignment correct (y = x shifted by 1)
```

---

## Training

### Pretraining

**Location**: `transformer_toolkit/trainer.py`

Full training loop for pretraining on raw text. Handles optimizer, learning rate schedule, gradient clipping, mixed precision, HuggingFace Hub integration, and graceful interruption.

#### TrainConfig

```python
from transformer_toolkit.trainer import TrainConfig

cfg = TrainConfig(
    # ── steps ─────────────────────────────────────────────────────────
    max_steps        = 10000,   # total optimizer steps
    eval_every       = 500,     # validation frequency
    save_every       = 1000,    # checkpoint frequency
    log_every        = 50,      # print loss every N steps
    interruptible    = True,    # Ctrl+C → clean checkpoint

    # ── optimiser ─────────────────────────────────────────────────────
    lr               = 3e-4,    # peak learning rate after warmup
    min_lr           = 3e-5,    # floor LR at end of cosine decay
    weight_decay     = 0.1,     # L2 penalty on 2D weights only
    beta1            = 0.9,     # AdamW β₁
    beta2            = 0.95,    # AdamW β₂
    grad_clip        = 1.0,     # max gradient norm

    # ── lr schedule ───────────────────────────────────────────────────
    warmup_steps     = 200,     # linear ramp from 0 to peak_lr

    # ── efficiency ────────────────────────────────────────────────────
    grad_accum_steps = 4,       # effective batch = batch_size × grad_accum
    mixed_precision  = True,    # automatic bf16/fp16 on CUDA
    grad_checkpoint  = False,   # recompute activations (~20% slower, 60% less VRAM)

    # ── checkpoints ───────────────────────────────────────────────────
    ckpt_dir         = "checkpoints",
    save_best        = True,    # save best.pt when val loss improves
    save_step_ckpts  = True,    # save step_N.pt every save_every steps

    # ── huggingface hub ───────────────────────────────────────────────
    hf_repo          = None,              # "username/model-name"
    hf_private       = True,
    hf_push_best     = True,    # push whenever val loss improves
    hf_push_every_n  = False,   # push every save_every steps
    hf_push_end      = True,    # push at end of training
    hf_push_on_pause = True,    # push on Ctrl+C
)
```

#### Training Loop

```python
from transformer_toolkit.trainer import Trainer

trainer = Trainer(
    model      = model,
    train_dl   = train_dl,
    val_dl     = val_dl,
    vocab_size = tok.vocab_size,
    cfg        = cfg,
    tokenizer  = tok,        # optional — used for Hub uploads
)

# Start training
trainer.train()

# Resume from checkpoint
trainer.train(resume_from="checkpoints/step_2000.pt")
```

**Example training output:**

```
⚡ Transformer Toolkit Trainer
steps=3000  lr=0.0003  warmup=200  accum=4
mixed_precision=True  grad_clip=1.0

step    100/3000  ████████░░░░░░░░░░░░░░░░  loss 3.1423  lr 1.5e-04  eta 4m
step    200/3000  ████████████░░░░░░░░░░░░  loss 2.8901  lr 3.0e-04  eta 3m

● eval  step 300  val_loss 2.7130  ppl 15.07  ▼0.1823  ★ best
```

**Expected loss curve (healthy run):**

| Step | Target val loss | Notes |
|------|-----------------|-------|
| init | ~log(vocab_size) | ~8.99 for vocab=8000 |
| 100 | 5-7 | Learning patterns |
| 300 | 3-5 | Confirm training works |
| 1000 | 2-3.5 | Good progress |
| 3000 | 1.5-2.5 | Typical small model |

If val loss > 8.0 at step 300 → initialization issue (check weight tying).  
If val loss < 1.0 before step 1000 → overfitting on small dataset.

---

### Supervised Fine-Tuning (SFT)

**Location**: `transformer_toolkit/sft_trainer.py`, `transformer_toolkit/sft_dataloader.py`

Full SFT pipeline — teach a pretrained model to follow instructions in a specific conversation format. Handles multi-turn conversations, loss masking, special tokens validation.

#### How SFT Works

During pretraining, the model learns language from raw text. SFT teaches it to follow a specific conversation format with proper roles, tokens, and stop conditions.

**Key idea**: Loss masking. Only the **assistant's response** contributes to loss:

```
<|start_header_id|>user<|end_header_id|>        → loss=0  (context)
What is Python?<|eot_id|>                        → loss=0
<|start_header_id|>assistant<|end_header_id|>   → loss=0  (header)

Python is a programming language.<|eot_id|>     → loss=1  (response)
[EOS]                                            → loss=1  (model learns to stop)
```

---

### Chat Templates

**Location**: `transformer_toolkit/chat_template.py`

Defines how conversations are formatted. Four presets available; pick one and use consistently.

#### Available Presets

| Preset | Format | Special tokens | Modern |
|--------|--------|----------------|--------|
| `llama3` | `<\|start_header_id\|>role<\|end_header_id\|>\n\ncontent<\|eot_id\|>` | IDs 9-11 | ✓ Recommended |
| `chatml` | `<\|im_start\|>role<\|im_end\|>\ncontent<\|im_end\|>` | IDs 7-8 | ✓ Popular |
| `gemma` | `<start_of_turn>role<end_of_turn>\ncontent<end_of_turn>` | IDs 12-13 | ✓ Supported |
| `alpaca` | `### Instruction:\ncontent\n\n### Response:\ncontent` | None | Older |
| `raw` | `User: content\nAssistant: content` | None | Fallback |

#### Using a Chat Template

```python
from transformer_toolkit.chat_template import ChatTemplate

template = ChatTemplate("llama3")

# Format messages for display/logging
msgs = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "A programming language."},
]
text, loss_mask_ranges = template.format_messages(msgs)
print(text)
```

#### Custom Template

```python
template = ChatTemplate(
    preset            = "chatml",
    assistant_header  = "<|im_start|>assistant\n",   # loss=0
    assistant_closer  = "<|im_end|>\n",              # loss=1
)
```

---

## Inference & Utilities

### Inference Engine

**Location**: `transformer_toolkit/inference.py`

High-level API for generation. Handles sampling parameters, streaming output, device selection.

#### InferenceConfig

```python
from transformer_toolkit.inference import InferenceConfig

cfg = InferenceConfig(
    max_new_tokens      = 200,      # max tokens to generate
    temperature         = 0.8,      # higher = random, lower = focused
    top_k               = 50,       # keep top-k tokens
    top_p               = 0.9,      # nucleus sampling
    repetition_penalty  = 1.1,      # penalize repeated tokens
    stream              = True,     # print tokens as they generate
    device              = "cuda",   # or "cpu"
)
```

#### Using Inference

```python
from transformer_toolkit.inference import Inference

inference = Inference(model, tok, cfg)

# Single generation
output = inference.generate(prompt="Once upon a time")

# Streaming output
inference.stream = True
output = inference.generate(prompt="Hello world")
```

---

### HuggingFace Hub Integration

**Location**: `transformer_toolkit/hf_hub.py`

Push and pull models to/from HuggingFace Hub. Automatic during training if configured.

#### Login

```python
from transformer_toolkit.hf_hub import login

login(token="hf_your_token_here")
```

#### Push to Hub

```python
from transformer_toolkit.hf_hub import push_to_hub

push_to_hub(
    repo_id   = "username/my-model",
    model     = model,
    cfg       = cfg_model,
    tokenizer = tok,
    metrics   = {"val_loss": 1.83, "perplexity": 6.23},
    step      = 3000,
    private   = True,
)
```

#### Pull from Hub

```python
from transformer_toolkit.hf_hub import pull_from_hub

pull_from_hub("username/my-model", save_dir="checkpoints")
# Downloads: model.pt, tokenizer.json, config.json, metrics.json
```

---

### Color Utilities

**Location**: `transformer_toolkit/colors.py`

Internal ANSI color codes for formatted console output. Used throughout the library for training logs, debug output, error messages.

```python
from transformer_toolkit.colors import C

print(f"{C.BOLD}{C.GREEN}Success!{C.RESET}")
print(f"{C.YELLOW}⚠  Warning{C.RESET}")
print(f"{C.RED}✗ Error{C.RESET}")
```

**Colors available**: `RED`, `GREEN`, `YELLOW`, `BLUE`, `CYAN`, `MAGENTA`, `WHITE`  
**Styles available**: `BOLD`, `DIM`, `RESET`

---

## Quick Start

## Examples

### Small Model — Shakespeare

Complete example training a small transformer on Shakespeare text (< 5 minutes on 4GB GPU).

```python
import torch, os
from transformer_toolkit.model import Transformer, TransformerConfig
from transformer_toolkit.c_tokenizers import RustBPETokenizer
from transformer_toolkit.dataloader import DataConfig, from_binary, from_npy_split, save_binary
from transformer_toolkit.trainer import Trainer, TrainConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer — train once, reuse
tok = RustBPETokenizer()
if os.path.exists("tokenizer.json"):
    tok.load("tokenizer.json")
else:
    tok.train(open("shakespeare.txt", encoding="utf-8").readlines(), vocab_size=8000)
    tok.save("tokenizer.json")

# data — tokenize once, reuse memmap splits
cfg_data = DataConfig(seq_len=128, batch_size=32, split=0.9, stride=None)
if os.path.exists("train.npy") and os.path.exists("val.npy"):
    train_dl, val_dl = from_npy_split("train.npy", "val.npy", cfg_data, tokenizer=tok)
else:
    if not os.path.exists("data.bin"):
        save_binary(tok.encode(open("shakespeare.txt", encoding="utf-8").read()), "data.bin")
    train_dl, val_dl = from_binary("data.bin", cfg_data,
                                    train_path="train.npy", val_path="val.npy",
                                    tokenizer=tok)

# model
model = Transformer(TransformerConfig(
    vocab_size  = tok.vocab_size,
    dim         = 384,
    n_layers    = 6,
    n_heads     = 6,
    n_kv_heads  = 3,
    attn        = "gqa",
    ffn         = "swiglu",
    hidden_dim  = 1536,
    norm        = "rmsnorm",
    pos_enc     = "rope",
    dropout     = 0.1,
    tie_weights = False,
)).to(DEVICE)
print(f"params: {model.n_params()}")   # ~15M

# train
trainer = Trainer(model, train_dl, val_dl, tok.vocab_size, TrainConfig(
    max_steps        = 3000,
    warmup_steps     = 200,
    eval_every       = 300,
    lr               = 3e-4,
    grad_accum_steps = 4,
    mixed_precision  = True,
    save_best        = True,
    save_step_ckpts  = True,
))
trainer.train()
```

### Large Dataset — HuggingFace Streaming

Stream a dataset without downloading it fully:

```python
from transformer_toolkit.dataloader import DataConfig, from_hf, from_npy_split
from transformer_toolkit.c_tokenizers import HFTokenizer

tok = HFTokenizer("HuggingFaceTB/SmolLM-135M")
cfg = DataConfig(seq_len=512, batch_size=16, stride=None, num_workers=4)

# first run — streams, tokenizes, caches as memmap splits
train_dl, val_dl = from_hf(
    dataset_name = "roneneldan/TinyStories",
    tokenizer    = tok,
    cfg          = cfg,
    bos_id       = tok._tok.bos_token_id,
    eos_id       = tok._tok.eos_token_id,
    train_path   = "train.npy",
    val_path     = "val.npy",
)

# future runs — zero download, memmap loads directly
train_dl, val_dl = from_npy_split("train.npy", "val.npy", cfg, tokenizer=tok)
```

### MoE Model — Sparse Experts

Train a mixture-of-experts model for parameter efficiency:

```python
model = Transformer(TransformerConfig(
    vocab_size     = tok.vocab_size,
    dim            = 512,
    n_layers       = 8,
    n_heads        = 8,
    attn           = "flash",
    ffn            = "moe",
    n_experts      = 8,
    top_k          = 2,
    moe_aux_weight = 0.01,
    pos_enc        = "rope",
    dropout        = 0.1,
    tie_weights    = False,
)).to("cuda")

# The Trainer adds aux_loss to loss automatically
trainer = Trainer(model, train_dl, val_dl, tok.vocab_size, TrainConfig(
    max_steps = 5000,
    lr        = 3e-4,
))
trainer.train()
```

### SFT on Pretrained Model

Fine-tune a pretrained model on instruction-following data:

```python
import torch
from transformer_toolkit import Transformer, TransformerConfig
from transformer_toolkit import SFTTrainer, TrainConfig
from transformer_toolkit import from_sft_json, SFTDataConfig
from transformer_toolkit.c_tokenizers import RustBPETokenizer

DEVICE = "cuda"

# Load pretrained tokenizer and model
tok = RustBPETokenizer()
tok.load("tokenizer.json")

model = Transformer(TransformerConfig(
    vocab_size=tok.vocab_size,
    dim=512, n_layers=8, n_heads=8, n_kv_heads=2,
    attn="gqa", ffn="swiglu", hidden_dim=2048,
    norm="rmsnorm", pos_enc="rope", max_seq=512,
)).to(DEVICE)

# Load pretrained checkpoint
ckpt = torch.load("pretraining_checkpoints/best.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model"])

# Prepare SFT data
cfg_sft = SFTDataConfig(
    tokenizer=tok, seq_len=512, batch_size=8, split=0.95,
    template="llama3", truncation_strategy="turn", debug=True
)
train_dl, val_dl = from_sft_json("instructions.jsonl", tok, cfg_sft)

# Fine-tune
trainer = SFTTrainer(
    model=model, train_dl=train_dl, val_dl=val_dl,
    vocab_size=tok.vocab_size,
    cfg=TrainConfig(
        max_steps=2000, lr=1e-4, warmup_steps=100,
        eval_every=100, save_every=200,
        save_best=True, ckpt_dir="sft_checkpoints",
    ),
    tokenizer=tok,
)
trainer.train()
```

---

## API Reference

### Model API

**Module**: `transformer_toolkit.model`

| Class/Function | Purpose |
|---|---|
| `TransformerConfig` | Dataclass controlling all architecture decisions |
| `Transformer` | Main model class — forward pass and generation |
| `TransformerBlock` | Single transformer block with gradient checkpointing |

**Key Methods**:

```python
# Transformer
model.forward(tokens: Tensor) → (logits, aux_loss)
model.generate(tokens, max_new, temperature, top_k, top_p) → Tensor
model.n_params() → str
model.debug_gradients() → None
model.debug_weights() → None
model.state_dict_for_save() → dict  # for weight-tied models
model.load_state_dict_with_tie(state_dict) → None  # for weight-tied models
```

### Attention API

**Module**: `transformer_toolkit.attention`

```python
from transformer_toolkit.attention import (
    MultiHeadAttention,
    GroupedQueryAttention,
    MultiQueryAttention,
    FlashAttention,
    MLAttention,
)

# All follow same interface
attn = MultiHeadAttention(dim, n_heads, pos_enc=...)(x)
```

### Feed-Forward API

**Module**: `transformer_toolkit.feed_forward`

```python
from transformer_toolkit.feed_forward import (
    FFN, ReLUFFN, GLU, ReGLU, GeGLU, SwiGLU,
    MoE, ExpertChoiceMoE, SharedExpertMoE,
)

# All follow same interface
ffn = SwiGLU(dim, hidden_dim)
output, aux_loss = ffn(x)
```

### Normalization API

**Module**: `transformer_toolkit.normalization`

```python
from transformer_toolkit.normalization import LayerNorm, RMSNorm, DeepNorm

norm = RMSNorm(dim, eps=1e-6)
x_normalized = norm(x)
```

### Positional Encoding API

**Module**: `transformer_toolkit.positional_encodings`

```python
from transformer_toolkit.positional_encodings import (
    SinusoidalPE, LearnedPE, RoPE, ALiBi
)

pe = RoPE(dim, max_seq=2048)
q_rotated, k_rotated = pe.rotate(q, k)
```

### Tokenizer API

**Module**: `transformer_toolkit.c_tokenizers`

```python
from transformer_toolkit.c_tokenizers import (
    ByteLevelTokenizer,
    RustBPETokenizer,
    HFTokenizer,
)

# All follow BaseTokenizer interface
tok.train(texts, vocab_size)
ids = tok.encode(text)
text = tok.decode(ids)
tok.save(path)
tok.load(path)
vocab_size = tok.vocab_size
```

### Dataloader API

**Module**: `transformer_toolkit.dataloader`

```python
from transformer_toolkit.dataloader import (
    DataConfig,
    from_binary,
    from_npy_split,
    from_files,
    from_hf,
    save_binary,
)

cfg = DataConfig(seq_len=512, batch_size=32)
train_dl, val_dl = from_binary("data.bin", cfg, tokenizer=tok)
```

### Trainer API

**Module**: `transformer_toolkit.trainer`

```python
from transformer_toolkit.trainer import Trainer, TrainConfig

cfg = TrainConfig(max_steps=10000, lr=3e-4)
trainer = Trainer(model, train_dl, val_dl, vocab_size, cfg, tokenizer=tok)
trainer.train()
trainer.train(resume_from="checkpoints/step_5000.pt")
```

### SFT API

**Module**: `transformer_toolkit.sft_trainer`, `transformer_toolkit.sft_dataloader`

```python
from transformer_toolkit import (
    SFTTrainer,
    SFTDataConfig,
    from_sft_strings,
    from_sft_json,
    from_sft_files,
    from_sft_hf,
    ChatTemplate,
)

cfg = SFTDataConfig(
    tokenizer=tok, seq_len=512, batch_size=8,
    template="llama3", truncation_strategy="turn"
)
train_dl, val_dl = from_sft_json("data.jsonl", tok, cfg)

trainer = SFTTrainer(model, train_dl, val_dl, vocab_size, cfg, tokenizer=tok)
trainer.train()
```

### Chat Template API

**Module**: `transformer_toolkit.chat_template`

```python
from transformer_toolkit.chat_template import ChatTemplate

template = ChatTemplate("llama3")  # or "chatml", "gemma", "alpaca", "raw"
text, loss_ranges = template.format_messages(messages)
template.validate_template()  # check special tokens
```

### Inference API

**Module**: `transformer_toolkit.inference`

```python
from transformer_toolkit.inference import Inference, InferenceConfig

cfg = InferenceConfig(
    max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.9
)
engine = Inference(model, tokenizer, cfg)
output = engine.generate(prompt="Hello")
```

### HuggingFace Hub API

**Module**: `transformer_toolkit.hf_hub`

```python
from transformer_toolkit.hf_hub import login, push_to_hub, pull_from_hub

login(token="...")
push_to_hub(repo_id="username/model", model=model, cfg=cfg, tokenizer=tok)
pull_from_hub("username/model", save_dir="checkpoints")
```

---

## Architecture Reference

Complete transformer architecture overview:

```
Input tokens                [B, T]
      │
      ▼
Embedding + DropOut
      │
      ▼  (SinusoidalPE or LearnedPE added here, if selected)
      │
      ▼  × n_layers
┌─────────────────────────────────────────────────┐
│  RMSNorm / LayerNorm                            │
│  ├─ Attention (MHA/GQA/MQA/Flash/MLA)          │
│  │  ├─ RoPE applied to q, k (if selected)      │
│  │  ├─ ALiBi bias added to scores (if sel.)    │
│  │  └─ Causal mask applied automatically       │
│  └─ + Residual connection                       │
│                                                 │
│  RMSNorm / LayerNorm                            │
│  ├─ FFN / SwiGLU / MoE                          │
│  └─ + Residual connection                       │
└─────────────────────────────────────────────────┘
      │
      ▼
Final RMSNorm / LayerNorm
      │
      ▼
Output Linear Head              [B, T, vocab_size]
      │
      ▼
Logits + Aux Loss (MoE only)
```

**Data Flow:**
1. Tokenize input text → token IDs: `[B, T]`
2. Embed tokens → embeddings: `[B, T, dim]`
3. Add positional encoding (if sinusoidal/learned)
4. Pass through n_layers transformer blocks (each applies attention + FFN)
5. Apply final norm
6. Linear projection to vocabulary → logits: `[B, T, vocab_size]`
7. Cross-entropy loss computed only on response tokens (SFT) or all tokens (pretraining)

---

## Requirements & Installation

### Core Requirements

| Package | Version | Purpose | Required? |
|---------|---------|---------|-----------|
| `torch` | ≥ 2.0 | PyTorch (GPU recommended) | ✓ Yes |
| `numpy` | any | Memmap dataloading | ✓ Yes |
| `pydantic` | any | Config validation | ✓ Yes |

### Optional Dependencies

| Package | Version | Purpose | Command |
|---------|---------|---------|---------|
| `tokenizers` | any | RustBPETokenizer | `pip install tokenizers` |
| `transformers` | any | HFTokenizer, HF datasets | `pip install transformers` |
| `datasets` | any | HF dataset streaming | `pip install datasets` |
| `huggingface_hub` | any | Hub push/pull | `pip install huggingface_hub` |
| `hf-transfer` | any | Faster Hub uploads | `pip install hf-transfer` |

### Quick Install

**Full installation (all features):**

```bash
pip install transformer-toolkit tokenizers transformers datasets huggingface_hub hf-transfer
```

**Minimal installation (PyTorch + core):**

```bash
pip install torch numpy pydantic
pip install transformer-toolkit
```

### GPU Setup

**NVIDIA CUDA (recommended):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**AMD ROCm:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

---

## Troubleshooting

### Model Training Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Loss stuck at `log(vocab_size)` | Weight tying + initialization | Disable tying or scale embedding by 0.02 |
| Loss NaN after few steps | Learning rate too high | Reduce `lr` (try `1e-4` for SFT) |
| OOM (out of memory) | Batch too large | Reduce `batch_size` or enable `grad_checkpoint` |
| Training slow | Missing optimizations | Enable `mixed_precision=True`, use `flash` attention |
| Validation loss plateaus | Underfitting | Increase model size or training steps |

### Data Loading Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Memory spike on first load | Full dataset in RAM | Use `stride=None`, enable memmap saving |
| Slow data loading | Python GIL contention | Increase `num_workers` |
| Tokenizer error in SFT | Missing special tokens | Retrain tokenizer before SFT |
| Very low mask ratio | seq_len too large | Lower `seq_len` to match data |

### Device Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA out of memory | Model/batch too large | Use mixed precision, gradient checkpointing, smaller batch |
| Wrong device placement | Tensor on CPU, model on GPU | Ensure `.to(device)` before forward pass |
| Slow on GPU | Using CPU tensors | Use `.cuda()` on inputs before model calls |

---

## Citation

If you use Transformer-Toolkit in your research, please cite:

```bibtex
@software{transformer_toolkit_2026,
  title={Transformer-Toolkit: A Production Modular Transformer Library},
  author={Barbade, Govind},
  year={2026},
  url={https://github.com/Barbade22/transformer-toolkit}
}
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
