# Transformer-Toolkit

<p align="center">
  <img src="images/image.png" alt="Transformer Toolkit Logo" width="600"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/transformer-toolkit/"><img src="https://img.shields.io/pypi/v/transformer-toolkit?color=cyan&style=flat-square" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/transformer-toolkit/"><img src="https://static.pepy.tech/badge/transformer-toolkit" alt="Downloads"/></a>
  <a href="https://github.com/Barbade22/transformer-toolkit/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Barbade22/transformer-toolkit?style=flat-square" alt="License"/></a>
</p>


A modular, from-scratch transformer library for training and experimenting with modern LLM architectures. Swap attention types, positional encodings, FFN variants, and normalization — all from a single config object.

```bash
pip install transformer-toolkit
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Model](#model)
  - [TransformerConfig](#transformerconfig)
  - [Transformer](#transformer)
  - [Weight Tying](#weight-tying)
  - [Debug Mode](#debug-mode)
- [Attention](#attention)
- [Feed-Forward Networks](#feed-forward-networks)
- [Positional Encodings](#positional-encodings)
- [Normalization](#normalization)
- [Dataloader](#dataloader)
  - [DataConfig](#dataconfig)
  - [Binary Files](#loading-from-a-binary-file)
  - [Memmap NPY](#memmap--loading-pre-split-npy-files)
  - [Text Files](#loading-from-text-files)
  - [HuggingFace](#loading-from-huggingface)
  - [Debug Samples](#dataloader-debug-mode)
- [Tokenizers](#tokenizers)
  - [ByteLevelTokenizer](#byteleveltokenizer)
  - [RustBPETokenizer](#rustbpetokenizer)
  - [HFTokenizer](#hftokenizer)
- [Trainer](#trainer)
  - [TrainConfig](#trainconfig)
  - [Training Loop](#training-loop)
- [HuggingFace Hub](#huggingface-hub)
- [Generation](#generation)
- [Full Examples](#full-examples)
  - [Small Model — Shakespeare](#small-model--shakespeare)
  - [Large Dataset — HuggingFace Streaming](#large-dataset--huggingface-streaming)
  - [MoE Model](#moe-model)
- [Architecture Reference](#architecture-reference)
- [Requirements](#requirements)

---

## Quick Start

```python
import torch
from transformer_toolkit.model import Transformer, TransformerConfig
from transformer_toolkit.c_tokenizers import RustBPETokenizer
from transformer_toolkit.dataloader import DataConfig, from_binary, save_binary
from transformer_toolkit.trainer import Trainer, TrainConfig

# tokenizer
tok = RustBPETokenizer()
tok.train(open("data.txt", encoding="utf-8").readlines(), vocab_size=8000)
tok.save("tokenizer.json")

# data
save_binary(tok.encode(open("data.txt", encoding="utf-8").read()), "data.bin")
train_dl, val_dl = from_binary("data.bin", DataConfig(seq_len=128, batch_size=32))

# model
model = Transformer(TransformerConfig(
    vocab_size  = tok.vocab_size,
    dim         = 512,
    n_layers    = 8,
    n_heads     = 8,
    pos_enc     = "rope",
    tie_weights = False,   # recommended for training from scratch
)).to("cuda")

# train
trainer = Trainer(model, train_dl, val_dl, tok.vocab_size, TrainConfig(max_steps=3000))
trainer.train()
```

---

## Model

### TransformerConfig

All architecture decisions live in one dataclass. Pass it to `Transformer()`.

```python
from transformer_toolkit.model import TransformerConfig

cfg = TransformerConfig(
    # ── core ──────────────────────────────────────────────────────────
    vocab_size = 32000,      # tokenizer vocabulary size
    dim        = 512,        # model embedding dimension
    n_layers   = 8,          # number of transformer blocks
    n_heads    = 8,          # number of attention heads
    max_seq    = 2048,       # maximum sequence length

    # ── attention ─────────────────────────────────────────────────────
    attn       = "gqa",      # "mha" | "gqa" | "mqa" | "flash" | "mla"
    n_kv_heads = 4,          # gqa only — number of key/value heads
    latent_dim = 64,         # mla only — latent compression dimension

    # ── feed-forward ──────────────────────────────────────────────────
    ffn        = "swiglu",   # "ffn" | "swiglu" | "moe"
    hidden_dim = 2048,       # FFN inner dimension (default: dim × 4)
    n_experts  = 8,          # moe only — number of experts
    top_k      = 2,          # moe only — experts activated per token
    moe_aux_weight = 0.01,   # moe load-balancing loss coefficient

    # ── normalization ─────────────────────────────────────────────────
    norm       = "rmsnorm",  # "rmsnorm" | "layernorm"
    eps        = 1e-6,

    # ── positional encoding ───────────────────────────────────────────
    pos_enc    = "rope",     # "rope" | "sinusoidal" | "learned" | "alibi" | "none"

    # ── regularisation ────────────────────────────────────────────────
    dropout    = 0.1,

    # ── output ────────────────────────────────────────────────────────
    tie_weights = False,     # share embedding and output projection weights
                             # see Weight Tying section before enabling
)
```

### Transformer

```python
from transformer_toolkit.model import Transformer

model = Transformer(cfg).to("cuda")

print(model.n_params())   # "30.21M"

# forward pass — returns (logits, aux_loss)
# aux_loss is non-zero only for MoE; always add it to your training loss
logits, aux_loss = model(tokens)   # tokens: [B, T]  →  logits: [B, T, vocab_size]

# generation
output = model.generate(
    tokens      = prompt_tokens,   # [B, T]
    max_new     = 200,
    temperature = 0.8,
    top_k       = 40,
)
```

### Weight Tying

Weight tying makes the embedding matrix and the output projection share the same tensor in memory. This reduces parameter count and can improve perplexity, but requires careful initialization.

> **Important:** `nn.Embedding` initializes weights with `N(0, 1)` — values around ±5. When the head shares these large weights, it produces logits of ±400 at initialization instead of the expected ±3, causing loss to start at ~346 instead of the correct ~`log(vocab_size)`. The model cannot recover from this initialization.

**Recommended approach — disable tying for training from scratch:**

```python
cfg = TransformerConfig(
    ...
    tie_weights = False,   # safe default for training from scratch
)
```

**If you want to enable tying**, scale down the embedding at initialization:

```python
model = Transformer(cfg).to("cuda")

if cfg.tie_weights:
    with torch.no_grad():
        model.embed.weight.mul_(0.02)   # bring logits into ±3 range
```

**Checkpoint save/load with tying enabled** — use the dedicated helpers to prevent the tie from breaking across save/load cycles:

```python
# saving
torch.save({"model": model.state_dict_for_save(), ...}, "checkpoint.pt")

# loading
model.load_state_dict_with_tie(ckpt["model"])
```

### Debug Mode

Pass `debug=True` to `Transformer()` to get a model summary at construction and a full forward pass trace.

```python
model = Transformer(cfg, debug=True).to("cuda")
model.debug = False   # turn off after inspecting — runs on every forward pass
```

**What it prints at construction:**

```
  🏗️  Model summary
  params             16.35M
  dim                384
  n_layers           6
  entropy check → should be > 90% of log(vocab_size) at init

  parameter breakdown:
  embed     ███░░░░░░░░░░░░░░░░░  3.07M  18.8%
  blocks    ████████████████░░░░  13.28M  81.2%
```

**What it prints per forward pass:**

```
  🔬 Forward pass debug
  tokens   [32, 128]  int64
  embed    [32, 128, 384]  float32  min=-4.84  mean=+0.00  max=+4.95
  block 0  residual update norm ratio: 0.133   ← healthy (0.01–2.0)
  logits   [32, 128, 8000]  float32  min=-2.97  mean=0.00  max=+2.98
  entropy  8.821 / max 8.99  (98.1% of uniform)   ← healthy at init
```

**Entropy at init should be above 90% of `log(vocab_size)`**. If it shows `-0.0%`, the logit scale is wrong — check the weight tying section above.

**Additional debug utilities:**

```python
# after loss.backward() — inspect gradient health per parameter
model.debug_gradients()

# any time — inspect weight statistics per parameter
model.debug_weights()
```

---

## Attention

Five attention variants, all swappable via `TransformerConfig.attn`.

| Value | Class | Used in |
|-------|-------|---------|
| `"mha"` | `MultiHeadAttention` | Original Transformer, BERT, GPT-2 |
| `"gqa"` | `GroupedQueryAttention` | LLaMA 3, Mistral |
| `"mqa"` | `MultiQueryAttention` | Falcon, early Gemini |
| `"flash"` | `FlashAttention` | Any model on PyTorch ≥ 2.0 |
| `"mla"` | `MLAttention` | DeepSeek-V2/V3 |

**RoPE** is applied inside attention to `q` and `k` after head-splitting — not to the residual stream. It is instantiated once and shared across all layers. The cos/sin cache is kept in `float32` regardless of model dtype to preserve precision.

**ALiBi** bias is computed once per forward pass and passed as an additive mask to every block.

**Causal masking** is applied automatically inside each attention module. You do not need to pass a mask for standard language model training.

### Example — Flash Attention

```python
cfg = TransformerConfig(
    dim     = 512,
    n_heads = 8,
    attn    = "flash",   # uses torch.nn.functional.scaled_dot_product_attention
)
```

### Example — Grouped Query Attention (LLaMA-style)

```python
cfg = TransformerConfig(
    dim        = 512,
    n_heads    = 8,
    attn       = "gqa",
    n_kv_heads = 2,   # 4 query heads share each kv head → 4x KV cache reduction
)
```

---

## Feed-Forward Networks

| Value | Class | Used in |
|-------|-------|---------|
| `"ffn"` | `FFN` | Original Transformer, BERT |
| `"swiglu"` | `SwiGLU` | LLaMA, Mistral, PaLM |
| `"moe"` | `MoE` | Mixtral, GPT-4 (rumoured) |

### MoE — Mixture of Experts

When using `ffn="moe"`, the model forward pass returns an auxiliary load-balancing loss that **must** be added to the main loss. Without it, all tokens collapse onto 1–2 experts within a few hundred steps and the remaining experts never get trained.

```python
cfg = TransformerConfig(
    ffn            = "moe",
    n_experts      = 8,
    top_k          = 2,
    moe_aux_weight = 0.01,   # weight of the load-balancing term (Mixtral uses 0.02)
)

logits, aux_loss = model(tokens)
ce_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
loss    = ce_loss + aux_loss   # aux_loss is 0.0 for non-MoE models — safe to always add
```

The `Trainer` handles `aux_loss` automatically — no changes to training code needed.

---

## Positional Encodings

| Value | Applied where | Notes |
|-------|---------------|-------|
| `"rope"` | Inside attention, on q and k | LLaMA, Mistral, Qwen — best for most use cases |
| `"sinusoidal"` | Residual stream before blocks | Original Transformer — no parameters |
| `"learned"` | Residual stream before blocks | BERT, GPT-2 — trainable |
| `"alibi"` | Additive bias on attention scores | Good for length generalization |
| `"none"` | Not applied | Bare model with no position information |

Each encoding applies exactly once in exactly one place — there is no double-application between the residual stream and attention.

---

## Normalization

| Value | Class | Notes |
|-------|-------|-------|
| `"rmsnorm"` | `RMSNorm` | LLaMA, Mistral, Qwen — no mean subtraction, no bias, faster |
| `"layernorm"` | `LayerNorm` | BERT, GPT-2 — classic formulation with bias |

---

## Dataloader

### DataConfig

```python
from transformer_toolkit.dataloader import DataConfig

cfg = DataConfig(
    seq_len     = 128,    # sequence length fed to the model
    batch_size  = 32,     # samples per batch
    split       = 0.9,    # fraction of data used for training
    stride      = None,   # None = non-overlapping windows (strongly recommended)
                          # stride < seq_len = overlapping windows (more samples,
                          # but causes rapid overfitting on small datasets)
    shuffle     = True,
    num_workers = 4,
    pin_memory  = True,
    debug       = False,  # print decoded sample preview before training starts
    debug_n     = 3,      # number of samples to show when debug=True
)
```

> **stride** — the default `stride=None` (equivalent to `stride=seq_len`) produces non-overlapping windows. For a 1.86M token dataset with `seq_len=128` this gives ~14,600 clean distinct samples. Setting `stride=1` gives 1.86M heavily-overlapping samples and causes rapid overfitting on small datasets.

### Loading from a Binary File

```python
from transformer_toolkit.dataloader import save_binary, from_binary

# tokenize once and save to disk
save_binary(tok.encode(text), "data.bin")

# load — supports both raw uint16 binary and .npy
train_dl, val_dl = from_binary("data.bin", cfg, tokenizer=tok)

# pass train_path and val_path to save splits as memmap .npy for future runs
train_dl, val_dl = from_binary(
    "data.bin", cfg,
    train_path = "train.npy",
    val_path   = "val.npy",
    tokenizer  = tok,
)
```

### Memmap — Loading Pre-split NPY Files

On second and subsequent runs, load the pre-split `.npy` files directly. The token file stays on disk — only the pages actually accessed are loaded into RAM. Scales to datasets of 100GB+.

```python
from transformer_toolkit.dataloader import from_npy_split

train_dl, val_dl = from_npy_split("train.npy", "val.npy", cfg, tokenizer=tok)
```

### Loading from Text Files

```python
from transformer_toolkit.dataloader import from_files

train_dl, val_dl = from_files(
    paths      = ["data1.txt", "data2.txt"],
    tokenizer  = tok,
    cfg        = cfg,
    train_path = "train.npy",   # optional — saves splits for future memmap reuse
    val_path   = "val.npy",
    bos_id     = tok.bos_id,    # optional — wrap each document with BOS/EOS tokens
    eos_id     = tok.eos_id,
)
```

### Loading from HuggingFace

```python
from transformer_toolkit.dataloader import from_hf

# streaming — no full download required, works with infinite datasets
cfg_stream = DataConfig(seq_len=512, batch_size=16, streaming=True)
train_dl, val_dl = from_hf("roneneldan/TinyStories", tok, cfg_stream)

# in-memory — downloads fully, then splits and optionally saves as .npy
train_dl, val_dl = from_hf(
    dataset_name = "roneneldan/TinyStories",
    tokenizer    = tok,
    cfg          = cfg,
    text_col     = "text",
    bos_id       = 1,
    eos_id       = 2,
    train_path   = "train.npy",
    val_path     = "val.npy",
)
```

### Dataloader Debug Mode

```python
cfg = DataConfig(seq_len=128, batch_size=32, debug=True, debug_n=3)
train_dl, val_dl = from_binary("data.bin", cfg, tokenizer=tok)
```

Prints before training starts, showing decoded text and verifying x/y alignment:

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

## Tokenizers

Three tokenizers with a unified interface.

```python
from transformer_toolkit.c_tokenizers import (
    ByteLevelTokenizer,
    RustBPETokenizer,
    HFTokenizer,
)
```

### ByteLevelTokenizer

Zero dependencies. Every byte is a token (vocab size fixed at 256). Works on any language or encoding out of the box.

```python
tok = ByteLevelTokenizer()
ids = tok.encode("Hello world")   # [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
txt = tok.decode(ids)             # "Hello world"
print(tok.vocab_size)             # 256
```

### RustBPETokenizer

BPE tokenizer backed by HuggingFace's Rust `tokenizers` library. Trains approximately 100x faster than a pure Python BPE implementation.

```bash
pip install tokenizers
```

```python
tok = RustBPETokenizer()
tok.train(open("data.txt").readlines(), vocab_size=8000)
tok.save("tokenizer.json")

# on subsequent runs — load instead of retraining
tok.load("tokenizer.json")

ids = tok.encode("Hello world")
txt = tok.decode(ids)
print(tok.vocab_size)   # 8000
```

### HFTokenizer

Thin wrapper around any HuggingFace pretrained tokenizer.

```bash
pip install transformers
```

```python
tok = HFTokenizer("gpt2")
ids = tok.encode("Hello world")
txt = tok.decode(ids)
print(tok.vocab_size)   # 50257
```

---

## Trainer

### TrainConfig

```python
from transformer_toolkit.trainer import TrainConfig

cfg = TrainConfig(
    # ── steps ─────────────────────────────────────────────────────────
    max_steps        = 10000,   # total number of optimizer steps
    eval_every       = 500,     # run validation every N steps
    save_every       = 1000,    # save step_N.pt every N steps
    log_every        = 50,      # print loss and lr every N steps
    interruptible    = True,    # Ctrl+C saves a clean checkpoint instead of crashing

    # ── optimiser ─────────────────────────────────────────────────────
    lr               = 3e-4,   # peak learning rate after warmup
    min_lr           = 3e-5,   # floor lr at end of cosine decay (typically lr / 10)
    weight_decay     = 0.1,    # L2 penalty on 2D weights — biases and norms excluded
    beta1            = 0.9,    # AdamW beta1
    beta2            = 0.95,   # AdamW beta2
    grad_clip        = 1.0,    # max gradient norm

    # ── lr schedule ───────────────────────────────────────────────────
    warmup_steps     = 200,    # linear ramp from 0 to peak lr over this many steps

    # ── efficiency ────────────────────────────────────────────────────
    grad_accum_steps = 4,      # effective batch = batch_size × grad_accum_steps
    mixed_precision  = True,   # bf16/fp16 on CUDA, float32 on CPU automatically
    grad_checkpoint  = False,  # recompute activations during backward (~20% slower,
                               # but reduces VRAM by ~60% for large models)

    # ── checkpoints ───────────────────────────────────────────────────
    ckpt_dir         = "checkpoints",
    save_best        = True,        # save best.pt whenever val loss improves
    save_step_ckpts  = True,        # save step_N.pt every save_every steps

    # ── huggingface hub ───────────────────────────────────────────────
    hf_repo          = "username/my-model",   # None to disable
    hf_private       = True,
    hf_push_best     = True,    # push to hub whenever best val loss improves
    hf_push_every_n  = False,   # push to hub every save_every steps
    hf_push_end      = True,    # push to hub at end of training
    hf_push_on_pause = True,    # push to hub on Ctrl+C pause
)
```

### Training Loop

```python
from transformer_toolkit.trainer import Trainer

trainer = Trainer(
    model      = model,
    train_dl   = train_dl,
    val_dl     = val_dl,
    vocab_size = tok.vocab_size,
    cfg        = cfg_train,
    tokenizer  = tok,        # optional — used for HuggingFace hub uploads
)

# start training
trainer.train()

# resume from a checkpoint
trainer.train(resume_from="checkpoints/step_2000.pt")
```

Training output:

```
  ⚡ Transformer Toolkit Trainer
  steps=3000  lr=0.0003  warmup=200  accum=4
  mixed_precision=True  grad_clip=1.0

  step    100/3000  ████████░░░░░░░░░░░░░░░░  loss 3.1423  lr 1.5e-04  eta 4m
  step    200/3000  ████████████░░░░░░░░░░░░  loss 2.8901  lr 3.0e-04  eta 3m

  ● eval  step 300  val_loss 2.7130  ppl 15.07  ▼0.1823  ★ best
```

**Expected loss curve for a healthy run:**

| Step | Expected val loss | Notes |
|------|-------------------|-------|
| 0 | ~`log(vocab_size)` | Random init — ~8.99 for vocab=8000 |
| 100 | 5–7 | Model learning basic patterns |
| 300 | 3–5 | First eval — confirm learning is happening |
| 1000 | 2–3.5 | Good progress |
| 3000 | 1.5–2.5 | Healthy final loss for a small model |

If val loss is still above 8.0 at step 300, something is wrong with initialization. If it drops below 1.0 before step 1000 on a small dataset, you are overfitting.

---

## HuggingFace Hub

### Login

```python
from transformer_toolkit.hf_hub import login

login(token="hf_your_token_here")
```

### Push to Hub

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

### Pull from Hub

```python
from transformer_toolkit.hf_hub import pull_from_hub

pull_from_hub("username/my-model", save_dir="checkpoints")
# downloads: model.pt, tokenizer.json, config.json, metrics.json
```

---

## Generation

```python
from transformer_toolkit.model import Transformer, TransformerConfig
from transformer_toolkit.c_tokenizers import RustBPETokenizer
from transformer_toolkit.trainer import load_ckpt
import torch

DEVICE = torch.device("cuda")

# load tokenizer — always load the saved file, never retrain
tok = RustBPETokenizer()
tok.load("tokenizer.json")

# model config must match the training config exactly
cfg = TransformerConfig(
    vocab_size  = tok.vocab_size,
    dim         = 384,
    n_layers    = 6,
    n_heads     = 6,
    attn        = "gqa",
    n_kv_heads  = 3,
    ffn         = "swiglu",
    hidden_dim  = 1536,
    norm        = "rmsnorm",
    pos_enc     = "rope",
    dropout     = 0.0,        # always 0.0 at inference
    tie_weights = False,
)
model = Transformer(cfg).to(DEVICE)
load_ckpt("checkpoints/best.pt", model)
model.eval()

def generate(prompt, max_new=200, temperature=0.8, top_k=40):
    ids    = tok.encode(prompt)
    tokens = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out    = model.generate(tokens, max_new=max_new,
                             temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())

print(generate("ROMEO:"))
print(generate("To be or not to be,"))
```

**Generation parameters:**

| Parameter | Effect | Recommended range |
|-----------|--------|-------------------|
| `temperature` | Higher = more random, lower = more repetitive | 0.7 – 1.0 |
| `top_k` | Only sample from the top-k most likely tokens | 20 – 50 |
| `max_new` | Number of new tokens to generate | 100 – 500 |

---

## Full Examples

### Small Model — Shakespeare

Suitable for any GPU. Trains in under 10 minutes on a 4GB card.

```python
import torch, os
from transformer_toolkit.model import Transformer, TransformerConfig
from transformer_toolkit.c_tokenizers import RustBPETokenizer
from transformer_toolkit.dataloader import DataConfig, from_binary, from_npy_split, save_binary
from transformer_toolkit.trainer import Trainer, TrainConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer — load if saved, train once otherwise
tok = RustBPETokenizer()
if os.path.exists("tokenizer.json"):
    tok.load("tokenizer.json")
else:
    tok.train(open("shakespeare.txt", encoding="utf-8").readlines(), vocab_size=8000)
    tok.save("tokenizer.json")

# data — tokenize once, reuse memmap splits on subsequent runs
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

```python
from transformer_toolkit.dataloader import DataConfig, from_hf, from_npy_split
from transformer_toolkit.c_tokenizers import HFTokenizer

tok = HFTokenizer("HuggingFaceTB/SmolLM-135M")
cfg = DataConfig(seq_len=512, batch_size=16, stride=None, num_workers=4)

# first run — downloads, tokenizes, and saves as memmap .npy splits
train_dl, val_dl = from_hf(
    dataset_name = "roneneldan/TinyStories",
    tokenizer    = tok,
    cfg          = cfg,
    bos_id       = tok._tok.bos_token_id,
    eos_id       = tok._tok.eos_token_id,
    train_path   = "train.npy",
    val_path     = "val.npy",
)

# second+ runs — zero RAM overhead, loads directly from disk
train_dl, val_dl = from_npy_split("train.npy", "val.npy", cfg, tokenizer=tok)
```

### MoE Model

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

# The Trainer adds aux_loss to ce_loss automatically — no changes needed
trainer = Trainer(model, train_dl, val_dl, tok.vocab_size, TrainConfig(
    max_steps = 5000,
    lr        = 3e-4,
))
trainer.train()
```

---

## Architecture Reference

```
Input tokens [B, T]
      │
      ▼
Embedding [B, T, dim]
      │
      ▼  SinusoidalPE or LearnedPE added here (if selected)
      │
      ▼  × n_layers
┌─────────────────────────────────────────────┐
│  RMSNorm / LayerNorm                        │
│  Attention  ← RoPE applied to q,k here     │
│             ← ALiBi bias added to scores   │
│  Residual connection                        │
│                                             │
│  RMSNorm / LayerNorm                        │
│  FFN / SwiGLU / MoE                        │
│  Residual connection                        │
└─────────────────────────────────────────────┘
      │
      ▼
Final RMSNorm / LayerNorm
      │
      ▼
Linear head [B, T, vocab_size]  →  logits
```

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Core — required |
| `numpy` | any | Memmap dataloader — required |
| `pydantic` | any | TrainConfig validation — required |
| `tokenizers` | any | `RustBPETokenizer` |
| `transformers` | any | `HFTokenizer` |
| `datasets` | any | `from_hf()` |
| `huggingface_hub` | any | Hub push/pull |
| `hf_transfer` | any | Faster hub uploads (optional) |

Install all optional dependencies at once:

```bash
pip install transformer-toolkit tokenizers transformers datasets huggingface_hub hf-transfer
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.