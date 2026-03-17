# trainer.py

import math
import sys
import time
import signal
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from pydantic import BaseModel, Field
from typing import Optional
from .hf_hub import HFSyncWorker


# ─── colors ───────────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m";  BOLD  = "\033[1m";  DIM    = "\033[2m"
    GREEN   = "\033[32m"; CYAN  = "\033[36m"; YELLOW = "\033[33m"
    BLUE    = "\033[34m"; RED   = "\033[31m"; WHITE  = "\033[37m"
    MAGENTA = "\033[35m"


def _bar(current: int, total: int, width: int = 28) -> str:
    filled = int(width * current / max(total, 1))
    return f"{C.CYAN}{'█' * filled}{'░' * (width - filled)}{C.RESET}"


def _loss_color(loss: float) -> str:
    if loss > 4.0:  return C.RED
    if loss > 2.5:  return C.YELLOW
    if loss > 1.5:  return C.GREEN
    return C.CYAN


def _header(cfg):
    w = 62
    print(f"\n{C.BOLD}{C.CYAN}{'─' * w}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  ⚡ Transformer Toolkit Trainer{C.RESET}")
    print(f"{C.DIM}  steps={cfg.max_steps}  lr={cfg.lr}  warmup={cfg.warmup_steps}  accum={cfg.grad_accum_steps}{C.RESET}")
    print(f"{C.DIM}  mixed_precision={cfg.mixed_precision}  grad_clip={cfg.grad_clip}{C.RESET}")
    if cfg.hf_repo:
        print(f"{C.DIM}  hf_repo={cfg.hf_repo}  push_best={cfg.hf_push_best}  push_every_n={cfg.hf_push_every_n}  push_end={cfg.hf_push_end}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'─' * w}{C.RESET}\n")


def _eval_line(step, val_loss, ppl, best_loss, saved):
    lc      = _loss_color(val_loss)
    delta   = val_loss - best_loss
    delta_s = f"{C.GREEN}▼{abs(delta):.4f}{C.RESET}" if delta < 0 else f"{C.DIM}▲{delta:.4f}{C.RESET}"
    trophy  = f"  {C.YELLOW}{C.BOLD}★ best{C.RESET}" if saved else ""
    print(
        f"\n  {C.BOLD}{C.BLUE}● eval{C.RESET}  step {step}"
        f"  val_loss {lc}{C.BOLD}{val_loss:.4f}{C.RESET}"
        f"  ppl {C.CYAN}{ppl:.2f}{C.RESET}"
        f"  {delta_s}{trophy}\n"
    )


def _ckpt_line(path):
    print(f"  {C.DIM}💾 saved → {path}{C.RESET}")


def _pause_line(step, path):
    print(f"\n  {C.YELLOW}{C.BOLD}⏸  paused at step {step}{C.RESET}")
    print(f"  {C.DIM}resume: trainer.train(resume_from='{path}'){C.RESET}\n")


def _done_line(best_loss, elapsed):
    print(f"\n{C.BOLD}{C.GREEN}  ✓ training complete{C.RESET}"
          f"  best_val_loss={C.CYAN}{C.BOLD}{best_loss:.4f}{C.RESET}"
          f"  time={C.BLUE}{elapsed/60:.1f}m{C.RESET}\n")


# ─── config ───────────────────────────────────────────────────────────────────

class TrainConfig(BaseModel):
    # ── steps ─────────────────────────────────────────────────────────────────
    max_steps:        int   = Field(10000, description="Total number of optimizer steps to train for")
    eval_every:       int   = Field(500,   description="Run validation every N steps")
    save_every:       int   = Field(1000,  description="Save a step checkpoint every N steps")
    log_every:        int   = Field(50,    description="Print loss and lr to console every N steps")
    interruptible:    bool  = Field(True,  description="Ctrl+C saves a clean checkpoint instead of crashing")

    # ── optimiser ─────────────────────────────────────────────────────────────
    lr:               float = Field(3e-4,  description="Peak learning rate after warmup")
    weight_decay:     float = Field(0.1,   description="L2 penalty applied to 2D weights only, not biases or norms")
    beta1:            float = Field(0.9,   description="AdamW beta1 — running mean of gradients")
    beta2:            float = Field(0.95,  description="AdamW beta2 — running mean of squared gradients")
    grad_clip:        float = Field(1.0,   description="Max gradient norm — prevents exploding gradients")

    # ── lr schedule ───────────────────────────────────────────────────────────
    warmup_steps:     int   = Field(200,   description="Linearly ramp lr from 0 to peak over this many steps")
    min_lr:           float = Field(3e-5,  description="Floor lr after cosine decay, typically lr / 10")

    # ── efficiency ────────────────────────────────────────────────────────────
    grad_accum_steps: int   = Field(1,     description="Accumulate gradients over N batches — effective batch = batch_size × grad_accum_steps")
    mixed_precision:  bool  = Field(True,  description="Use bf16 or fp16 for faster training, falls back to fp32 on CPU")
    grad_checkpoint:  bool  = Field(False, description="Recompute activations during backward to save memory at ~20% speed cost")

    # ── checkpoints ───────────────────────────────────────────────────────────
    ckpt_dir:         str   = Field("checkpoints", description="Directory to save all checkpoint files")
    save_best:        bool  = Field(True,           description="Save best.pt whenever val loss improves")
    save_step_ckpts:  bool  = Field(True,           description="Save step_N.pt every save_every steps")

    # ── huggingface hub ───────────────────────────────────────────────────────
    hf_repo:          Optional[str] = Field(None,  description="HuggingFace repo id e.g. 'username/my-model'. None = disabled")
    hf_private:       bool          = Field(True,  description="Make HF repo private")
    hf_push_best:     bool          = Field(True,  description="Push to HF whenever best val loss improves")
    hf_push_every_n:  bool          = Field(False, description="Push to HF every save_every steps")
    hf_push_end:      bool          = Field(True,  description="Push to HF at end of training")
    hf_push_on_pause: bool          = Field(True,  description="Push to HF when training is paused with Ctrl+C")

    class Config:
        validate_assignment = True


# ─── lr schedule ──────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    # start from a small non-zero lr at step 0 (1/warmup_steps of peak)
    # so the very first update is not a zero-gradient no-op
    if step < cfg.warmup_steps:
        return cfg.lr * max(step, 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ─── checkpoint ───────────────────────────────────────────────────────────────

def save_ckpt(path: str, model, optimizer, scaler, step: int, val_loss: float):
    from pathlib import Path
    # FIX: use Path.parent.mkdir so an empty dirname (path with no directory)
    # doesn't cause os.makedirs("") to raise FileNotFoundError
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Use state_dict_for_save if available to avoid broken weight tying on resume
    model_sd = model.state_dict_for_save() if hasattr(model, "state_dict_for_save") else model.state_dict()
    torch.save({
        "step":      step,
        "val_loss":  val_loss,
        "model":     model_sd,
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
    }, path)
    _ckpt_line(path)


def load_ckpt(path: str, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Use load_state_dict_with_tie if available to re-apply weight tying
    if hasattr(model, "load_state_dict_with_tie"):
        model.load_state_dict_with_tie(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    if optimizer: optimizer.load_state_dict(ckpt["optimizer"])
    if scaler:    scaler.load_state_dict(ckpt["scaler"])
    print(f"  {C.GREEN}▶  resumed from step {ckpt['step']}{C.RESET}  best_loss={ckpt['val_loss']:.4f}\n")
    return ckpt["step"], ckpt["val_loss"]


# ─── eval ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_dl, vocab_size: int, device, cfg: TrainConfig) -> float:
    model.eval()
    dtype  = _dtype(cfg, device)
    losses = []
    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        with autocast(device_type=device.type, dtype=dtype):
            logits, _ = model(x)
            loss      = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.item())
        if len(losses) >= 20: break
    model.train()
    # FIX: guard against empty val_dl instead of dividing by zero
    return sum(losses) / len(losses) if losses else float("inf")


# ─── trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, train_dl, val_dl, vocab_size: int, cfg: TrainConfig, tokenizer=None):
        self.model      = model
        self.train_dl   = train_dl
        self.val_dl     = val_dl
        self.vocab_size = vocab_size
        self.cfg        = cfg
        self.tokenizer  = tokenizer
        self.device     = next(model.parameters()).device
        # FIX: pass device so _dtype() can return float32 on CPU
        self.dtype      = _dtype(cfg, self.device)
        self.best_loss  = float("inf")

        # gradient checkpointing — PyTorch native via block-level checkpoint.
        # gradient_checkpointing_enable() is a HuggingFace method and does not
        # exist on plain nn.Module. We set use_checkpoint on each block instead.
        if cfg.grad_checkpoint:
            for block in model.blocks:
                block.use_checkpoint = True

        # optimiser — no weight decay on biases/norms
        decay    = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() <  2]
        self.optimizer = torch.optim.AdamW([
            {"params": decay,    "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

        self.scaler     = GradScaler(enabled=cfg.mixed_precision and self.device.type != "cpu")
        self._pause_req = False

        # background HF upload worker — only created if hf_repo is set
        self._hf = HFSyncWorker() if cfg.hf_repo else None

        # pause handler
        if cfg.interruptible:
            signal.signal(signal.SIGINT,  self._handle_pause)
            signal.signal(signal.SIGTERM, self._handle_pause)

    def _handle_pause(self, signum, frame):
        print(f"\n{C.YELLOW}  ⏸  pause requested — finishing step...{C.RESET}")
        self._pause_req = True

    def train(self, resume_from: str = None):
        step     = 0
        # initialise val_loss so it is always defined even if save_every
        # fires before the first eval (i.e. save_every < eval_every)
        val_loss = float("inf")

        if resume_from:
            step, self.best_loss = load_ckpt(resume_from, self.model, self.optimizer, self.scaler)

        cfg       = self.cfg
        model     = self.model
        optimizer = self.optimizer
        loader    = _infinite(self.train_dl)

        _header(cfg)
        model.train()

        t_start = time.time()
        t0      = time.time()
        tokens  = 0

        print(f"  {C.DIM}starting...{C.RESET}  {_bar(0, cfg.max_steps)}  {C.DIM}step 0/{cfg.max_steps}{C.RESET}\n")

        while step < cfg.max_steps:

            # ── pause check ───────────────────────────────────
            if self._pause_req:
                path = f"{cfg.ckpt_dir}/pause_step_{step}.pt"
                save_ckpt(path, model, optimizer, self.scaler, step, val_loss)
                _pause_line(step, path)

                if self._hf and cfg.hf_repo and cfg.hf_push_on_pause:
                    print(f"  {C.YELLOW}⬆  pushing to hub before exit...{C.RESET}")
                    self._hf.push(
                        model     = model,
                        optimizer = optimizer,
                        scaler    = self.scaler,
                        val_loss  = val_loss,
                        repo_id   = cfg.hf_repo,
                        cfg       = model.cfg,
                        tokenizer = self.tokenizer,
                        metrics   = {"val_loss": val_loss, "step": step},
                        step      = step,
                        private   = cfg.hf_private,
                    )
                    # hf_hub.HFSyncWorker.wait() takes no arguments
                    self._hf.wait()
                    self._hf.shutdown()

                sys.exit(0)

            # ── lr update ─────────────────────────────────────
            lr = get_lr(step, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ── forward + backward ────────────────────────────
            optimizer.zero_grad()
            # FIX: accumulate loss over all micro-batches for correct logging
            accum_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                x, y = next(loader)
                x, y = x.to(self.device), y.to(self.device)
                with autocast(device_type=self.device.type, dtype=self.dtype):
                    logits, aux_loss = model(x)
                    ce_loss  = F.cross_entropy(
                        logits.view(-1, self.vocab_size), y.view(-1)
                    )
                    loss = (ce_loss + aux_loss) / cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()
                tokens += x.numel()

            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()

            step += 1

            # ── live progress bar (every step) ────────────────
            # FIX: accum_loss already equals mean loss (each micro loss was /grad_accum)
            current_loss = accum_loss
            dt           = time.time() - t0
            tps          = tokens / max(dt, 1e-6)
            eta          = (cfg.max_steps - step) * (time.time() - t_start) / max(step, 1)
            eta_str      = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
            lc           = _loss_color(current_loss)

            print(
                f"\r  {C.DIM}step{C.RESET} {C.BOLD}{C.WHITE}{step:>6}{C.RESET}/{cfg.max_steps}"
                f"  {_bar(step, cfg.max_steps)}"
                f"  {C.DIM}loss{C.RESET} {lc}{C.BOLD}{current_loss:.4f}{C.RESET}"
                f"  {C.DIM}lr{C.RESET} {C.MAGENTA}{lr:.1e}{C.RESET}"
                f"  {C.DIM}eta{C.RESET} {C.BLUE}{eta_str}{C.RESET}   ",
                end="", flush=True
            )

            if step % cfg.log_every == 0:
                print(f"  {C.DIM}tok/s{C.RESET} {C.YELLOW}{tps:,.0f}{C.RESET}")
                tokens = 0
                t0     = time.time()

            # ── eval ──────────────────────────────────────────
            if step % cfg.eval_every == 0:
                val_loss  = evaluate(model, self.val_dl, self.vocab_size, self.device, cfg)
                ppl       = math.exp(min(val_loss, 20))
                saved     = val_loss < self.best_loss
                prev_best = self.best_loss  # capture before mutating so delta is non-zero on new best

                if saved:
                    self.best_loss = val_loss
                    if cfg.save_best:
                        save_ckpt(f"{cfg.ckpt_dir}/best.pt", model, optimizer, self.scaler, step, val_loss)

                    if self._hf and cfg.hf_push_best:
                        self._hf.push(
                            model     = model,
                            optimizer = optimizer,
                            scaler    = self.scaler,
                            val_loss  = val_loss,
                            repo_id   = cfg.hf_repo,
                            cfg       = model.cfg,
                            tokenizer = self.tokenizer,
                            metrics   = {"val_loss": val_loss, "perplexity": ppl},
                            step      = step,
                            private   = cfg.hf_private,
                        )

                _eval_line(step, val_loss, ppl, prev_best, saved)
                # reset throughput counters so eval wall time doesn't
                # pollute the tok/s reading on the next log line
                tokens = 0
                t0     = time.time()

            # ── step checkpoint ───────────────────────────────
            if cfg.save_step_ckpts and step % cfg.save_every == 0:
                save_ckpt(f"{cfg.ckpt_dir}/step_{step}.pt", model, optimizer, self.scaler, step, val_loss)

                if self._hf and cfg.hf_push_every_n:
                    self._hf.push(
                        model   = model,
                        optimizer = optimizer,
                        scaler    = self.scaler,
                        val_loss  = val_loss,
                        repo_id = cfg.hf_repo,
                        cfg     = model.cfg,
                        metrics = {"val_loss": val_loss},
                        step    = step,
                        private = cfg.hf_private,
                    )

        # ── end of training ───────────────────────────────────
        _done_line(self.best_loss, time.time() - t_start)

        if self._hf and cfg.hf_push_end:
            self._hf.push(
                model     = model,
                optimizer = optimizer,
                scaler    = self.scaler,
                val_loss  = val_loss,
                repo_id   = cfg.hf_repo,
                cfg       = model.cfg,
                tokenizer = self.tokenizer,
                metrics   = {"val_loss": self.best_loss},
                step      = step,
                private   = cfg.hf_private,
            )

        if self._hf:
            self._hf.shutdown()


# ─── helpers ──────────────────────────────────────────────────────────────────

def _infinite(dl):
    while True:
        yield from dl


def _dtype(cfg: TrainConfig, device: torch.device) -> torch.dtype:
    # FIX: CPU does not support float16 autocast — always return float32 on CPU
    if not cfg.mixed_precision or device.type == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
