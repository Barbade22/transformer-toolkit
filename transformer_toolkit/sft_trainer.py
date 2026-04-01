# sft_trainer.py

"""
SFTTrainer — Supervised Fine-Tuning trainer.

Subclasses Trainer and overrides only the two methods that differ:
    _compute_loss()  : masked cross-entropy (response tokens only)
    _evaluate()      : masked validation loss

Everything else — optimizer, scaler, LR schedule, gradient clipping,
checkpointing, HF sync, Ctrl+C handler — is inherited unchanged.

Usage:
    from transformer_toolkit.sft_trainer import SFTTrainer
    from transformer_toolkit.trainer import TrainConfig

    trainer = SFTTrainer(
        model      = model,
        train_dl   = train_dl,   # yields (x, y, loss_mask)
        val_dl     = val_dl,
        vocab_size = tok.vocab_size,
        cfg        = TrainConfig(...),
        tokenizer  = tok,
    )
    trainer.train()

The only difference from the base Trainer is that train_dl / val_dl must
yield 3-tuples: (x, y, loss_mask). Use sft_dataloader.py to build them.
"""

import math
import time
import sys
import torch
import torch.nn.functional as F
from torch.amp import autocast

from .trainer import Trainer, TrainConfig, save_ckpt, get_lr, evaluate
from .trainer import _bar, _loss_color, _eval_line, _done_line, _header, _infinite, _dtype, C


# ─── masked loss helpers ──────────────────────────────────────────────────────

def masked_cross_entropy(
    logits:    torch.Tensor,   # (B, T, V)
    targets:   torch.Tensor,   # (B, T)
    mask:      torch.Tensor,   # (B, T)  float, 1=response 0=prompt
    vocab_size: int,
) -> torch.Tensor:
    """
    Cross-entropy loss computed only over response tokens.

    Numerically equivalent to reduction='mean' over unmasked positions.
    Returns a scalar tensor. Gradient flows only through masked=1 positions.

    If mask is all zeros (shouldn't happen after SFTDataset filtering, but
    just in case), falls back to standard mean CE to avoid division by zero.
    """
    B, T, V = logits.shape
    flat_logits  = logits.view(-1, vocab_size)   # (B*T, V)
    flat_targets = targets.view(-1)              # (B*T,)
    flat_mask    = mask.view(-1)                 # (B*T,)

    per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')  # (B*T,)

    n_response = flat_mask.sum().clamp(min=1.0)
    loss = (per_token_loss * flat_mask).sum() / n_response
    return loss


@torch.no_grad()
def evaluate_sft(
    model,
    val_dl,
    vocab_size: int,
    device,
    cfg: TrainConfig,
    max_batches: int = 20,
) -> float:
    """
    Validation loss using masked CE. Mirrors trainer.evaluate() in structure.
    """
    model.eval()
    dtype  = _dtype(cfg, device)
    losses = []

    for batch in val_dl:
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y       = batch
            mask       = torch.ones_like(y, dtype=torch.float)

        x, y, mask = x.to(device), y.to(device), mask.to(device)

        with autocast(device_type=device.type, dtype=dtype):
            logits, _ = model(x)
            loss      = masked_cross_entropy(logits, y, mask, vocab_size)

        losses.append(loss.item())
        if len(losses) >= max_batches:
            break

    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ─── SFTTrainer ───────────────────────────────────────────────────────────────

class SFTTrainer(Trainer):
    """
    Drop-in replacement for Trainer when fine-tuning with SFT data.

    Differences from base Trainer:
    - train_dl / val_dl must yield (x, y, loss_mask) 3-tuples
    - Loss is computed only on response tokens (mask == 1)
    - Validation uses the same masked loss
    - Prints token-level statistics (% response tokens per batch) when log_every fires

    Everything else is identical to Trainer.
    """

    def train(self, resume_from: str = None):
        step     = 0
        val_loss = float("inf")

        if resume_from:
            step, self.best_loss = self._load(resume_from)

        cfg       = self.cfg
        model     = self.model
        optimizer = self.optimizer
        loader    = _infinite(self.train_dl)

        _header(cfg)
        model.train()

        t_start     = time.time()
        t0          = time.time()
        tokens      = 0
        resp_tokens = 0   # response tokens seen (what the loss actually trains on)

        print(f"  {C.DIM}SFT mode — loss computed on response tokens only{C.RESET}")
        print(f"  {C.DIM}starting...{C.RESET}  {_bar(0, cfg.max_steps)}  "
              f"{C.DIM}step 0/{cfg.max_steps}{C.RESET}\n")

        while step < cfg.max_steps:

            # ── pause check ───────────────────────────────────
            if self._pause_req:
                path = f"{cfg.ckpt_dir}/pause_step_{step}.pt"
                save_ckpt(path, model, optimizer, self.scaler, step, val_loss)
                from .trainer import _pause_line
                _pause_line(step, path)

                if self._hf and cfg.hf_repo and cfg.hf_push_on_pause:
                    print(f"  {C.YELLOW}⬆  pushing to hub before exit...{C.RESET}")
                    self._hf.push(
                        model=model, optimizer=optimizer, scaler=self.scaler,
                        val_loss=val_loss, repo_id=cfg.hf_repo, cfg=model.cfg,
                        tokenizer=self.tokenizer,
                        metrics={"val_loss": val_loss, "step": step},
                        step=step, private=cfg.hf_private,
                    )
                    self._hf.wait()
                    self._hf.shutdown()
                sys.exit(0)

            # ── lr update ─────────────────────────────────────
            lr = get_lr(step, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # ── forward + backward ────────────────────────────
            optimizer.zero_grad()
            accum_loss  = 0.0
            accum_resp  = 0

            for _ in range(cfg.grad_accum_steps):
                batch = next(loader)

                # support both (x, y) and (x, y, mask) batches
                if len(batch) == 3:
                    x, y, mask = batch
                else:
                    x, y       = batch
                    mask       = torch.ones_like(y, dtype=torch.float)

                x, y, mask = (x.to(self.device),
                              y.to(self.device),
                              mask.to(self.device))

                with autocast(device_type=self.device.type, dtype=self.dtype):
                    logits, aux_loss = model(x)
                    ce_loss = masked_cross_entropy(logits, y, mask, self.vocab_size)
                    loss    = (ce_loss + aux_loss) / cfg.grad_accum_steps

                self.scaler.scale(loss).backward()
                accum_loss += loss.item()
                tokens     += x.numel()
                accum_resp += int(mask.sum().item())

            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()

            step        += 1
            resp_tokens += accum_resp

            # ── live progress bar ─────────────────────────────
            current_loss = accum_loss
            dt           = time.time() - t0
            tps          = tokens / max(dt, 1e-6)
            eta          = (cfg.max_steps - step) * (time.time() - t_start) / max(step, 1)
            eta_str      = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
            lc           = _loss_color(current_loss)
            resp_pct     = 100.0 * accum_resp / max(x.numel() * cfg.grad_accum_steps, 1)

            print(
                f"\r  {C.DIM}step{C.RESET} {C.BOLD}{C.WHITE}{step:>6}{C.RESET}/{cfg.max_steps}"
                f"  {_bar(step, cfg.max_steps)}"
                f"  {C.DIM}loss{C.RESET} {lc}{C.BOLD}{current_loss:.4f}{C.RESET}"
                f"  {C.DIM}resp%{C.RESET} {C.MAGENTA}{resp_pct:.1f}{C.RESET}"
                f"  {C.DIM}lr{C.RESET} {C.MAGENTA}{lr:.1e}{C.RESET}"
                f"  {C.DIM}eta{C.RESET} {C.BLUE}{eta_str}{C.RESET}   ",
                end="", flush=True
            )

            if step % cfg.log_every == 0:
                print(f"  {C.DIM}tok/s{C.RESET} {C.YELLOW}{tps:,.0f}{C.RESET}")
                tokens      = 0
                resp_tokens = 0
                t0          = time.time()

            # ── eval ──────────────────────────────────────────
            if step % cfg.eval_every == 0:
                val_loss  = evaluate_sft(model, self.val_dl, self.vocab_size,
                                         self.device, cfg)
                ppl       = math.exp(min(val_loss, 20))
                saved     = val_loss < self.best_loss
                prev_best = self.best_loss

                if saved:
                    self.best_loss = val_loss
                    if cfg.save_best:
                        save_ckpt(f"{cfg.ckpt_dir}/best.pt", model, optimizer,
                                  self.scaler, step, val_loss)
                    if self._hf and cfg.hf_push_best:
                        self._hf.push(
                            model=model, optimizer=optimizer, scaler=self.scaler,
                            val_loss=val_loss, repo_id=cfg.hf_repo, cfg=model.cfg,
                            tokenizer=self.tokenizer,
                            metrics={"val_loss": val_loss, "perplexity": ppl},
                            step=step, private=cfg.hf_private,
                        )

                _eval_line(step, val_loss, ppl, prev_best, saved)
                tokens      = 0
                resp_tokens = 0
                t0          = time.time()

            # ── step checkpoint ───────────────────────────────
            if cfg.save_step_ckpts and step % cfg.save_every == 0:
                save_ckpt(f"{cfg.ckpt_dir}/step_{step}.pt", model, optimizer,
                          self.scaler, step, val_loss)
                if self._hf and cfg.hf_push_every_n:
                    self._hf.push(
                        model=model, optimizer=optimizer, scaler=self.scaler,
                        val_loss=val_loss, repo_id=cfg.hf_repo, cfg=model.cfg,
                        metrics={"val_loss": val_loss},
                        step=step, private=cfg.hf_private,
                    )

        # ── end of training ───────────────────────────────────
        _done_line(self.best_loss, time.time() - t_start)

        if self._hf and cfg.hf_push_end:
            self._hf.push(
                model=model, optimizer=optimizer, scaler=self.scaler,
                val_loss=val_loss, repo_id=cfg.hf_repo, cfg=model.cfg,
                tokenizer=self.tokenizer,
                metrics={"val_loss": self.best_loss},
                step=step, private=cfg.hf_private,
            )

        if self._hf:
            self._hf.shutdown()

    # ── private helper (avoids importing load_ckpt's print side-effects) ──────

    def _load(self, path: str) -> tuple[int, float]:
        from .trainer import load_ckpt
        step, val_loss, _ = load_ckpt(path, self.model, self.optimizer, self.scaler)
        return step, val_loss