# example_sft.py
"""
Complete SFT training examples.

Steps:
  1. Pick a tokenizer            (section: tokenizer)
  2. Pick a chat template        (section: chat template)
  3. Configure SFTDataConfig     (section: data config)   ← debug= lives here
  4. Pick a data source          (section: options A–D)
  5. Build or load a model       (section: model)
  6. Configure TrainConfig       (section: train config)
  7. Run SFTTrainer              (section: trainer)
  8. Chat with the trained model (section: generation)

Debug mode (cfg_data.debug=True):
  Fires automatically after the dataloader is built, before training starts.
  For each of debug_n samples it prints:
    • token counts  — total / prompt / response / padding / resp%
    • mask bar      — 50-char visual  █=response  ░=prompt  ▒=mixed
    • turn breakdown— every contiguous prompt/response run with index range
    • formatted view— full decoded text, cyan=prompt  green=response
    • prompt blob   — decoded prompt tokens only (clipped)
    • response blob — decoded response tokens only (clipped)
    • sanity checks — mask integrity, x/y alignment, padding %, resp% ratio
  Pass tokenizer= to the loader to enable decoded text views.
  Set debug=False (or remove) once the pipeline is verified.
"""

import torch
from transformer_toolkit.model import Transformer, TransformerConfig
from transformer_toolkit.c_tokenizers import RustBPETokenizer
from transformer_toolkit.sft_dataloader import (
    SFTDataConfig, ChatTemplate,
    from_sft_json, from_sft_hf, from_sft_strings, from_sft_files,
)
from transformer_toolkit.sft_trainer import SFTTrainer
from transformer_toolkit.trainer import TrainConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── tokenizer ─────────────────────────────────────────────────────────────────
# Option 1: your own trained BPE tokenizer
tok = RustBPETokenizer()
tok.load("tokenizer.json")

# Option 2: any HuggingFace tokenizer
# from transformer_toolkit.c_tokenizers import HFTokenizer
# tok = HFTokenizer("mistralai/Mistral-7B-v0.1")

# Option 3: zero-dependency byte-level tokenizer (vocab_size=256, no training needed)
# from transformer_toolkit.c_tokenizers import ByteLevelTokenizer
# tok = ByteLevelTokenizer()


# ── chat template ──────────────────────────────────────────────────────────────
# Preset options: "chatml" | "llama3" | "alpaca" | "raw"
#
# chatml  (default, works well for models trained from scratch):
#   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n
#
# llama3  (matches Meta's Llama-3 instruct format):
#   <|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>...
#
# alpaca  (classic instruction format):
#   ### Instruction:\n{prompt}\n\n### Response:\n{response}
#
# raw     (plain newlines, good for simple byte-level or small-vocab models):
#   User: {prompt}\nAssistant: {response}
#
# Custom — override any format string while keeping the rest of a preset:
#
# template = ChatTemplate(
#     preset           = "chatml",
#     user_fmt         = "<|im_start|>user\n{content}<|im_end|>\n",
#     assistant_fmt    = "<|im_start|>assistant\n{content}<|im_end|>\n",
#     assistant_header = "<|im_start|>assistant\n",   # prefix before response text
# )

template = ChatTemplate("chatml")


# ── SFT data config ────────────────────────────────────────────────────────────
cfg_data = SFTDataConfig(
    seq_len     = 256,       # max tokens per sample (prompt + response combined)
    batch_size  = 4,
    split       = 0.9,       # fraction used for training; remainder → validation
    num_workers = 0,         # 0 = safe on Windows; increase on Linux/Mac
    pin_memory  = True,
    template    = template,
    schema      = "auto",    # auto-detect from keys, or force:
                             #   "prompt_response" | "messages" | "instruction"

    # ── debug mode ──────────────────────────────────────────────────────────
    # Fires once after the dataloader is built, before training.
    # Shows debug_n samples with full color-coded template view + sanity checks.
    # Always pass tokenizer= to your loader when debug=True — without it you
    # only get raw token IDs and the decoded views are skipped.
    debug   = True,
    debug_n = 2,             # how many samples to inspect (keep small, e.g. 2–5)
)


# ── OPTION A: local JSON or JSONL file ────────────────────────────────────────
# Schema is auto-detected from the first record.
# Supported schemas (all auto-detected, no schema= override needed):
#   {"prompt": "...", "response": "..."}
#   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
#   {"instruction": "...", "input": "...", "output": "..."}   (Alpaca)
#
# train_dl, val_dl = from_sft_json("my_data.json",  tok, cfg_data)
# train_dl, val_dl = from_sft_json("my_data.jsonl", tok, cfg_data)
#
# Force schema if auto-detect is wrong for your file:
# cfg_data.schema = "instruction"
# train_dl, val_dl = from_sft_json("alpaca_data.json", tok, cfg_data)


# ── OPTION B: HuggingFace dataset ─────────────────────────────────────────────
# Schema is auto-detected from column names.
#
# In-memory — downloads fully then tokenises (best for datasets < ~1M samples):
# train_dl, val_dl = from_sft_hf("tatsu-lab/alpaca",             tok, cfg_data)
# train_dl, val_dl = from_sft_hf("OpenAssistant/oasst1",         tok, cfg_data)
# train_dl, val_dl = from_sft_hf("HuggingFaceH4/ultrachat_200k", tok, cfg_data)
# train_dl, val_dl = from_sft_hf("teknium/OpenHermes-2.5",       tok, cfg_data)
#
# Streaming — tokenises on the fly, no RAM limit:
# cfg_data.streaming = True
# train_dl, val_dl = from_sft_hf("tiiuae/falcon-refinedweb", tok, cfg_data)
#
# Note: streaming mode uses first batch_size*20 rows as validation.
# Use in-memory mode for exact train/val splits.


# ── OPTION C: multiple local files ────────────────────────────────────────────
# Concatenates all files into one dataset, then splits train/val.
# Streaming mode splits files into train/val groups instead of concatenating.
#
# In-memory:
# train_dl, val_dl = from_sft_files(
#     ["data/alpaca.json", "data/oasst.jsonl", "data/custom.json"],
#     tok, cfg_data,
# )
#
# Streaming:
# cfg_data.streaming = True
# train_dl, val_dl = from_sft_files(
#     ["data/part1.jsonl", "data/part2.jsonl", "data/part3.jsonl"],
#     tok, cfg_data,
# )


# ── OPTION D: in-memory dicts — quick experiments and unit tests ───────────────
# All three schemas can be mixed in one list — auto-detect runs per-sample.
samples = [
    # prompt/response schema
    {"prompt": "What is 2+2?",
     "response": "2+2 equals 4."},

    {"prompt": "Name the planets of the solar system.",
     "response": "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune."},

    # multi-turn messages schema
    {"messages": [
        {"role": "user",      "content": "Tell me a joke."},
        {"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side!"},
        {"role": "user",      "content": "Give me another one."},
        {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything."},
    ]},

    # multi-turn with system prompt
    {"messages": [
        {"role": "system",    "content": "You are a helpful coding assistant."},
        {"role": "user",      "content": "How do I reverse a list in Python?"},
        {"role": "assistant", "content": "Use slicing: `my_list[::-1]` returns a reversed copy, "
                                         "or call `my_list.reverse()` to reverse in place."},
        {"role": "user",      "content": "What about a tuple?"},
        {"role": "assistant", "content": "Tuples are immutable, so use slicing: `my_tuple[::-1]`. "
                                         "This returns a new tuple."},
    ]},

    # Alpaca instruction schema (with optional input field)
    {"instruction": "Translate to French.", "input": "Hello, how are you?",
     "output": "Bonjour, comment allez-vous ?"},

    {"instruction": "Write a haiku about autumn.", "input": "",
     "output": "Crimson leaves descend,\nWind carries the fading warmth,\nSilence holds the frost."},

] * 30   # repeat so there are enough samples for a meaningful train/val split

train_dl, val_dl = from_sft_strings(samples, tok, cfg_data)
# ↑ pass tokenizer=tok as last arg to enable decoded text in debug output:
# train_dl, val_dl = from_sft_strings(samples, tok, cfg_data)
# (tokenizer is already the second positional arg — debug decoding is automatic)


# ── model ─────────────────────────────────────────────────────────────────────
cfg_model = TransformerConfig(
    vocab_size  = tok.vocab_size,
    dim         = 384,
    n_layers    = 2,
    n_heads     = 2,
    attn        = "gqa",
    n_kv_heads  = 3,
    ffn         = "swiglu",
    hidden_dim  = 1536,
    norm        = "rmsnorm",
    pos_enc     = "rope",
    dropout     = 0.1,
    tie_weights = False,
)
model = Transformer(cfg_model).to(DEVICE)

# Fine-tuning from a CLM pre-trained checkpoint (recommended workflow):
#   1. Pre-train with the standard Trainer on raw text  → checkpoints/best.pt
#   2. Load that checkpoint here, then SFT on instruction data
#
# from transformer_toolkit.trainer import load_ckpt
# load_ckpt("checkpoints/best.pt", model)

print(f"model: {model.n_params()}")


# ── train config ──────────────────────────────────────────────────────────────
# SFT uses a lower LR than CLM pre-training (typically lr / 3 to lr / 10).
# Fewer steps are needed — the model already knows the language, it's learning
# the response format and instruction-following behaviour.
cfg_train = TrainConfig(
    max_steps        = 1000,
    warmup_steps     = 50,
    eval_every       = 100,
    save_every       = 200,
    log_every        = 25,
    lr               = 1e-4,       # pre-training used 3e-4 → SFT uses 1e-4
    min_lr           = 1e-5,
    grad_accum_steps = 4,          # effective batch = batch_size * grad_accum_steps
    mixed_precision  = True,
    grad_checkpoint  = False,
    save_best        = True,
    save_step_ckpts  = True,
    ckpt_dir         = "sft_checkpoints",
    hf_repo          = None,       # "username/my-sft-model" to push to HF Hub
    hf_private       = True,
    hf_push_best     = True,
    hf_push_end      = True,
)


# ── SFTTrainer ────────────────────────────────────────────────────────────────
# Drop-in replacement for Trainer. Only difference: train_dl/val_dl must yield
# (x, y, loss_mask) 3-tuples — which from_sft_* loaders produce automatically.
#
# The progress bar shows an extra  resp%  column — the fraction of tokens in
# each step that are response tokens (the ones loss is actually computed on).
# Low resp% (< 10%) means prompts are very long; consider shorter prompts or
# a larger seq_len.
trainer = SFTTrainer(
    model      = model,
    train_dl   = train_dl,
    val_dl     = val_dl,
    vocab_size = tok.vocab_size,
    cfg        = cfg_train,
    tokenizer  = tok,
)
trainer.train()

# Resume from a saved checkpoint:
# trainer.train(resume_from="sft_checkpoints/best.pt")


# ── generation / inference after SFT ──────────────────────────────────────────
def chat(
    prompt:      str,
    system:      str   = None,
    max_new:     int   = 200,
    temperature: float = 0.8,
    top_k:       int   = 50,
) -> str:
    """
    Generate a response using the trained model.

    Uses the same ChatTemplate as training so the model sees the exact token
    sequences it was trained on. Appends the assistant header to prime generation
    and stops at the first end-of-turn token if present in the decoded output.

    Args:
        prompt      : user message
        system      : optional system prompt (prepended if provided)
        max_new     : maximum new tokens to generate
        temperature : sampling temperature (lower = more deterministic)
        top_k       : top-k sampling (1 = greedy)

    Returns:
        decoded response string (assistant turn only)
    """
    tpl  = template   # reuse the same template used during training

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    # format all turns, then append the assistant header to prime generation
    full_text, _ = tpl.format_messages(msgs)
    primed       = full_text + tpl.assistant_header

    ids = tok.encode(primed)
    x   = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    out = model.generate(x, max_new=max_new, temperature=temperature, top_k=top_k)

    new_ids  = out[0][len(ids):].tolist()
    response = tok.decode(new_ids)

    # strip trailing end-of-turn markers if the model emits them
    for eos_marker in ("<|im_end|>", "<|eot_id|>", "</s>"):
        if eos_marker in response:
            response = response[:response.index(eos_marker)]

    return response.strip()


# Single-turn example:
# print(chat("What is the capital of France?"))

# With system prompt:
# print(chat(
#     prompt = "How do I reverse a string in Python?",
#     system = "You are a concise coding assistant. Answer in 1-2 sentences.",
# ))

# Multi-turn (manual history management — model is stateless):
# history = []
# while True:
#     user_input = input("You: ").strip()
#     if not user_input: break
#     history.append({"role": "user", "content": user_input})
#     full, _ = template.format_messages(history)
#     primed  = full + template.assistant_header
#     ids     = tok.encode(primed)
#     x       = torch.tensor([ids[-cfg_model.max_seq:]], dtype=torch.long).to(DEVICE)
#     out     = model.generate(x, max_new=200, temperature=0.8, top_k=50)
#     reply   = tok.decode(out[0][len(ids):].tolist()).split("<|im_end|>")[0].strip()
#     history.append({"role": "assistant", "content": reply})
#     print(f"Assistant: {reply}\n")