# test_sft.py
"""
Minimal SFT smoke test — no files, no HF, no GPU required.
Trains a tiny model on in-memory sample data for 30 steps.
If it runs without errors and val_loss prints, everything is wired correctly.
"""

import torch
from transformer_toolkit.model import Transformer, TransformerConfig
from transformer_toolkit.c_tokenizers import RustBPETokenizer
from transformer_toolkit.sft_dataloader import (
    SFTDataConfig, ChatTemplate,
    from_sft_strings,
)
from transformer_toolkit.sft_trainer import SFTTrainer
from transformer_toolkit.trainer import TrainConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")


# ── tokenizer — train tiny BPE on the samples themselves ─────────────────────
tok = RustBPETokenizer()
tok.train([
    "What is 2+2? 2+2 equals 4.",
    "Name a planet. Mars is a planet.",
    "What color is the sky? The sky is blue.",
    "What is the capital of France? The capital of France is Paris.",
    "How do you reverse a list in Python? Use my_list[::-1].",
    "Tell me a joke. Why did the chicken cross the road? To get to the other side.",
    "What is water made of? Water is made of hydrogen and oxygen, H2O.",
    "Who wrote Hamlet? William Shakespeare wrote Hamlet.",
], vocab_size=512)
print(f"tokenizer vocab_size: {tok.vocab_size}")


# ── sample data — all three schemas mixed ────────────────────────────────────
samples = [
    # prompt/response schema
    {"prompt": "What is 2+2?",
     "response": "2+2 equals 4."},
    {"prompt": "Name a planet.",
     "response": "Mars is a planet."},
    {"prompt": "What color is the sky?",
     "response": "The sky is blue."},
    {"prompt": "What is the capital of France?",
     "response": "The capital of France is Paris."},

    # messages schema (multi-turn)
    {"messages": [
        {"role": "user",      "content": "How do you reverse a list in Python?"},
        {"role": "assistant", "content": "Use my_list[::-1] to reverse a list."},
        {"role": "user",      "content": "What about a tuple?"},
        {"role": "assistant", "content": "Use my_tuple[::-1] — tuples are immutable so it returns a new one."},
    ]},
    {"messages": [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side."},
    ]},

    # instruction/output schema (Alpaca)
    {"instruction": "Explain what water is made of.", "input": "",
     "output": "Water is made of hydrogen and oxygen, H2O."},
    {"instruction": "Who wrote Hamlet?", "input": "",
     "output": "William Shakespeare wrote Hamlet."},

] * 10  # repeat so train/val split has enough samples on both sides


# ── data config ───────────────────────────────────────────────────────────────
cfg_data = SFTDataConfig(
    seq_len    = 64,           # short — matches tiny model max_seq
    batch_size = 4,
    split      = 0.9,
    template   = ChatTemplate("chatml"),
    schema     = "auto",
    debug      = True,
    debug_n    = 2,
)

train_dl, val_dl = from_sft_strings(samples, tok, cfg_data)


# ── tiny model ────────────────────────────────────────────────────────────────
# Rules: n_kv_heads must divide n_heads evenly
#        hidden_dim must be > dim
#        max_seq must match seq_len
cfg_model = TransformerConfig(
    vocab_size  = tok.vocab_size,
    dim         = 64,
    n_layers    = 2,
    n_heads     = 2,
    n_kv_heads  = 1,       # GQA 2:1  (2 query heads share 1 kv head)
    attn        = "gqa",
    ffn         = "swiglu",
    hidden_dim  = 128,
    norm        = "rmsnorm",
    pos_enc     = "rope",
    dropout     = 0.0,
    tie_weights = False,
    max_seq     = 64,
)

model = Transformer(cfg_model, debug=True).to(DEVICE)
model.debug = False
print(f"model params: {model.n_params()}")


# ── single forward pass sanity check ─────────────────────────────────────────
print("\n── forward pass check ──")
model.debug = True
x, y, mask = next(iter(train_dl))
with torch.no_grad():
    logits, aux = model(x.to(DEVICE))
print(f"logits: {list(logits.shape)}  (expect [batch, seq, vocab])")
print(f"aux   : {aux}")
model.debug = False


# ── train config — 30 steps, no checkpoints, no HF ───────────────────────────
cfg_train = TrainConfig(
    max_steps        = 30,
    warmup_steps     = 5,
    eval_every       = 10,
    save_every       = 9999,   # effectively disabled
    log_every        = 5,
    lr               = 3e-4,
    min_lr           = 3e-5,
    grad_accum_steps = 1,
    mixed_precision  = False,  # off — cleaner output on CPU
    grad_checkpoint  = False,
    save_best        = False,
    save_step_ckpts  = False,
    interruptible    = False,
    hf_repo          = None,
)


# ── SFTTrainer ────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model      = model,
    train_dl   = train_dl,
    val_dl     = val_dl,
    vocab_size = tok.vocab_size,
    cfg        = cfg_train,
    tokenizer  = tok,
)
trainer.train()

print("\n── smoke test passed ✓ ──")