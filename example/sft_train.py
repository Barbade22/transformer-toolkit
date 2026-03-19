# test_conversation.py

from transformer_toolkit.c_tokenizers import RustBPETokenizer
from transformer_toolkit.chat_template import ChatTemplate
from transformer_toolkit.sft_dataloader import SFTDataConfig, from_sft_strings
from transformer_toolkit.model import TransformerConfig, Transformer
from transformer_toolkit.trainer import TrainConfig, Trainer
from transformer_toolkit.sft_trainer import SFTTrainer
# ── tokenizer ─────────────────────────────────────────────────────────────────
tok = RustBPETokenizer()
# tok.train(
#     texts=[
#         "What is Python? Python is a programming language.",
#         "How do you reverse a list? Use my_list[::-1].",
#         "What is a tuple? A tuple is an immutable sequence.",
#         "How do you open a file? Use open() with a context manager.",
#         "What is a dictionary? A dict stores key-value pairs.",
#     ],
#     vocab_size=512,
# )
# tok.load("big_tokenizer.json")
# print(f"vocab_size: {tok.vocab_size}")

# # verify special tokens
# for s in ["<|im_start|>", "<|im_end|>"]:
#     ids = tok.encode(s)
#     print(f"  {s!r} → id {ids[0]}  {'✓' if len(ids)==1 else 'BROKEN'}")


# ── sample data ───────────────────────────────────────────────────────────────
samples = [

    # single-turn
    {"messages": [
        {"role": "user",      "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]},

    # multi-turn
    {"messages": [
        {"role": "user",      "content": "How do you reverse a list in Python?"},
        {"role": "assistant", "content": "Use my_list[::-1] to reverse a list."},
        {"role": "user",      "content": "What about a tuple?"},
        {"role": "assistant", "content": "Use my_tuple[::-1] — it returns a new tuple since tuples are immutable."},
    ]},

    # multi-turn with system prompt
    {"messages": [
        {"role": "system",    "content": "You are a helpful Python tutor."},
        {"role": "user",      "content": "How do I open a file?"},
        {"role": "assistant", "content": "Use open() with a context manager: with open('file.txt') as f: data = f.read()"},
        {"role": "user",      "content": "What mode do I use for writing?"},
        {"role": "assistant", "content": "Pass mode='w' to open() for writing, or 'a' to append."},
    ]},

] * 10  # repeat so train/val split has enough samples
tok.load("big_tokenizer.json")

# # ── config ────────────────────────────────────────────────────────────────────
cfg = SFTDataConfig(
    tokenizer            = tok,
    seq_len              = 128,
    batch_size           = 2,
    split                = 0.9,
    template             = "llama3",
    schema               = "auto",
    truncation_strategy  = "turn",
    debug                = False,
    debug_n              = 3,
)

train_dl, val_dl = from_sft_strings(samples, tok, cfg)

# # ── check a batch ─────────────────────────────────────────────────────────────
# print("\n── batch check ──")
# x, y, mask = next(iter(train_dl))
# print(f"x    shape : {list(x.shape)}")
# print(f"y    shape : {list(y.shape)}")
# print(f"mask shape : {list(mask.shape)}")
# print(f"mask sum   : {mask.sum().item():.0f} response tokens in batch")

# # decode first sample to verify template looks right
# print("\n── first sample decoded ──")
# print(tok.decode(x[0].tolist()))

# from transformer_toolkit.c_tokenizers import RustBPETokenizer
# from transformer_toolkit.dataloader import DataConfig, from_binary, save_binary
# from transformer_toolkit.chat_template import ChatTemplate
# tok = RustBPETokenizer()

# # option 1 — read whole file as one string, split into lines
# with open("input2.txt", encoding="utf-8") as f:
#     lines = f.read().splitlines()

# tok.train(texts = lines, vocab_size=100_000,pre_tokenizer="bytelevel")
# # tok.load("big_tokenizer.json")
# print(f"vocab_size: {tok.vocab_size}")
# print(f"bos_id: {tok.bos_id}")
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cdf_model = TransformerConfig(
    vocab_size  = tok.vocab_size,
    dim         = 384,
    n_layers    = 2,
    n_heads     = 2,
    attn        = "gqa",
    n_kv_heads  = 1,
    ffn         = "swiglu",
    hidden_dim  = 1536,
    norm        = "rmsnorm",
    pos_enc     = "rope",
    dropout     = 0.1,
    tie_weights = False,
    max_seq=128
)

model = Transformer(cdf_model).to(DEVICE)

cfg_train = TrainConfig(
    max_steps        = 5000,
    warmup_steps     = 500,
    eval_every       = 500,
    save_every       = 200,
    log_every        = 100,
    lr               = 1e-4,       # pre-training used 3e-4 → SFT uses 1e-4
    min_lr           = 1e-6,
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


trainer = SFTTrainer(
    model      = model,
    train_dl   = train_dl,
    val_dl     = val_dl,
    vocab_size = tok.vocab_size,
    cfg        = cfg_train,
    tokenizer  = tok,
)
trainer.train()
