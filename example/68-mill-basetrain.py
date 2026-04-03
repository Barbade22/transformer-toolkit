import torch
import os
from datasets import load_dataset
from transformer_toolkit.model import Transformer, TransformerConfig
from transformer_toolkit.c_tokenizers import RustBPETokenizer
from transformer_toolkit.dataloader import DataConfig, from_hf
from transformer_toolkit.trainer import Trainer, TrainConfig
from transformer_toolkit.hf_hub import login

# ── login ──
login(token="hf_Your-Hf token")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

# ── tokenizer — train on TinyStories, not input.txt ──
TOK_PATH = "fine_tokenizer.json"
tok = RustBPETokenizer()

if os.path.exists(TOK_PATH):
    tok.load(TOK_PATH)
    print(f"tokenizer loaded — vocab_size={tok.vocab_size}")
else:
    print("streaming 500k docs for tokenizer training...")
    ds      = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    samples = []
    for row in ds:
        samples.append(row["text"])
        if len(samples) % 100000 == 0:
            print(f"  {len(samples):,} docs")
        if len(samples) >= 500000:
            break
    tok.train(samples, vocab_size=32000)
    tok.save(TOK_PATH)
    print(f"tokenizer trained — vocab_size={tok.vocab_size}")

# ── data ──
cfg_data = DataConfig(
    seq_len    = 512,
    batch_size = 108,
    debug      = False,
    debug_n    = 2,
    streaming  = True,
)

hf_kwargs = dict(
    dataset_name = "HuggingFaceFW/fineweb",
    tokenizer    = tok,
    cfg          = cfg_data,
    split        = "train",
    config       = "sample-10BT",
)
train_dl, val_dl = from_hf(**hf_kwargs) 

# ── model — 20M ──
cfg_model = TransformerConfig(
    vocab_size  = tok.vocab_size,
    dim         = 512,
    n_layers    = 12,
    n_heads     = 8,
    n_kv_heads  = 2,
    ffn         = "swiglu",
    hidden_dim  = 1536,
    norm        = "rmsnorm",
    pos_enc     = "rope",
    dropout     = 0.1,
    tie_weights = False,
    max_seq     = 512,
)

model = Transformer(cfg_model).to(DEVICE)
print(f"model: {model.n_params()}")

# ── train ──
cfg_train = TrainConfig(
    max_steps        = 80000,
    warmup_steps     = 200,
    eval_every       = 1000,
    save_every       = 1000,
    log_every        = 50,
    lr     = 5e-6,
    min_lr = 1e-6,
    grad_accum_steps = 2,
    mixed_precision  = True,
    grad_checkpoint  = False,
    save_best        = True,
    save_step_ckpts  = True,
    interruptible    = True,
    hf_repo          = "Govind222/Scratch",
    hf_private       = True,
    hf_push_best     = True,
    hf_push_every_n  = False,
    hf_push_end      = True,
    hf_push_on_pause = True,
)

trainer = Trainer(
    model,
    train_dl,
    val_dl,
    tok.vocab_size,
    cfg_train,
    tokenizer = tok,
    hf_kwargs = hf_kwargs,
)

# trainer.train()
# trainer.train(resume_from="checkpoints/pause_step_4101.pt")
trainer.train(resume_from="checkpoints/best.pt")