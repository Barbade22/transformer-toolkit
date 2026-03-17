import torch
import os
from model import Transformer, TransformerConfig
from c_tokenizers import RustBPETokenizer
from dataloader import DataConfig, from_binary, save_binary
from trainer import Trainer, TrainConfig
from hf_hub import login

# ── login ──
login(token="hf_Your_token")   # full token from huggingface.co/settings/tokens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

# ── tokenizer ──
tok = RustBPETokenizer()
tok.train(open("input.txt").readlines(), vocab_size=8000)

# ── data ── tokenize once, reuse binary
if not os.path.exists("data.bin"):
    save_binary(tok.encode(open("input.txt").read()), "data.bin")

cfg_data         = DataConfig(
    seq_len    = 128,   # enough context to learn structure
    batch_size = 32,    # much faster than 1
    split      = 0.9,
)
train_dl, val_dl = from_binary("data.bin", cfg_data)

# ── model ──
cfg_model = TransformerConfig(
    vocab_size = tok.vocab_size,   # ~8000
    dim        = 512,              # smaller model for this dataset size
    n_layers   = 6,
    n_heads    = 8,
    attn       = "gqa",            # mla needs more tuning — gqa is stable
    n_kv_heads = 4,
    ffn        = "swiglu",
    hidden_dim = 512,              # hidden should be > dim
    norm       = "rmsnorm",
    pos_enc    = "rope",
)
model = Transformer(cfg_model).to(DEVICE)
print(f"model: {model.n_params()}")

# ── train ──
cfg_train = TrainConfig(
    max_steps        = 5000,
    warmup_steps     = 200,
    eval_every       = 500,
    save_every       = 1000,
    log_every        = 50,
    lr               = 3e-4,
    min_lr           = 3e-5,
    grad_accum_steps = 4,          # effective batch = 32 * 4 = 128
    mixed_precision  = True,
    grad_checkpoint  = False,
    save_best        = True,
    save_every_n     = True,
    interruptible    = True,

    hf_repo          = "Govind222/Test-shakespeare",
    hf_private       = True,
    hf_push_best     = True,
    hf_push_every_n  = False,
    hf_push_end      = True,
    hf_push_on_pause = True,
)

trainer = Trainer(model, train_dl, val_dl, tok.vocab_size, cfg_train, tokenizer=tok)
trainer.train()