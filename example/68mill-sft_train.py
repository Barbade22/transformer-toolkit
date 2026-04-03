from signal import valid_signals
import torch
import os
from datasets import load_dataset
from transformer_toolkit.model import Transformer, TransformerConfig
from transformer_toolkit.c_tokenizers import RustBPETokenizer
from transformer_toolkit.sft_dataloader import SFTDataConfig, from_sft_strings
from transformer_toolkit.sft_trainer import SFTTrainer
from transformer_toolkit.trainer import TrainConfig
from transformer_toolkit.hf_hub import login
from torch.utils.data import DataLoader
import pickle
# ── login ──
login(token="hf_Your-Hf token")

from huggingface_hub import hf_hub_download

# ckpt_path = hf_hub_download(
#     repo_id  = "Govind222/Test-shakespeare",
#     filename = "checkpoint.pt",
# )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

# ── tokenizer ──
tok = RustBPETokenizer()
tok.load("fine_tokenizer.json")
print(f"tokenizer loaded — vocab_size={tok.vocab_size}")

# ── sft data ──
cfg_sft = SFTDataConfig(
    tokenizer  = tok,
    seq_len    = 512,
    batch_size = 96,
    split      = 0.9,
    shuffle    = True,
    debug      = True,
    debug_n    = 10,
    template   = "chatml",
    schema     = "auto",
    truncation_strategy = "tail",
)

import json
# ds      = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")
with open("mes_v3.json", "r", encoding="utf-8") as f:
    data = json.load(f)
with open("mes.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)
# # samples = []
# # data = data[:200]
data = data2 + data
samples = []

for row in data:
    msgs = row["messages"]

    for i in range(len(msgs)):
        if msgs[i]["role"] == "assistant":
            # STRICT slice — stops exactly at assistant
            sample_msgs = msgs[:i+1]

            # ensure valid structure
            if sample_msgs[0]["role"] != "user":
                continue

            samples.append({
                "messages": sample_msgs
            })
# print(f"conversations: {len(samples):,}")
print(f"conversations: {len(samples):,}")
train_dl, val_dl = from_sft_strings(samples, tok, cfg_sft)
# ── cache tokenized dataloaders ──
# CACHE_PATH = "sft_cache_v2_seq512.pkl"

# if os.path.exists(CACHE_PATH):
#     print("loading cache...")
#     with open(CACHE_PATH, "rb") as f:
#         tr_dataset, vl_dataset = pickle.load(f)
#     train_dl = DataLoader(tr_dataset, batch_size=96, shuffle=True,  pin_memory=True)
#     val_dl   = DataLoader(vl_dataset, batch_size=96, shuffle=False, pin_memory=True)
#     print("cache loaded ✓")
#     print(len(train_dl))
#     print(len(val_dl))
# else:
# train_dl, val_dl = from_sft_strings(samples, tok, cfg_sft)
# with open(CACHE_PATH, "wb") as f:
#     pickle.dump((train_dl.dataset, val_dl.dataset), f)
# print("cache saved ✓")
# ── model ──

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
    dropout     = 0.0,
    tie_weights = False,
    max_seq     = 512,
)
model = Transformer(cfg_model).to(DEVICE)
print(f"model: {model.n_params()}")

# ── sft train config ──
cfg_train = TrainConfig(
    max_steps        = 37000 + 20000,
    warmup_steps     = 0,
    lr               = 1e-5,
    min_lr           = 1e-6,
    eval_every       = 500,
    save_every       = 1000,
    log_every        = 50,
    grad_accum_steps = 2,
    mixed_precision  = True,
    save_best        = True,
    save_step_ckpts  = True,
    interruptible    = True,
    hf_repo          = "Govind222/ScratchSFT2",
    hf_private       = True,
    hf_push_best     = True,
    hf_push_end      = True,
    hf_push_on_pause = True,
)

trainer = SFTTrainer(
    model,
    train_dl,
    val_dl,
    tok.vocab_size,
    cfg_train,
    tokenizer = tok,
)

# trainer.train(resume_from=ckpt_path)
# trainer.train(resume_from="checkpoints/best.pt")
trainer.train(resume_from="checkpoints/step_37000.pt")