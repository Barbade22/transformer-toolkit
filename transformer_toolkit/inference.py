import torch
import torch.nn.functional as F
from pathlib import Path
from .colors import C, _section, _info, _ok, _err


# ─── inference config ─────────────────────────────────────────────────────────

class InferenceConfig:
    def __init__(
        self,
        max_new_tokens: int   = 200,    # max tokens to generate
        temperature:    float = 0.8,    # higher = more random, lower = more focused
        top_k:          int   = 50,     # keep only top k tokens at each step
        top_p:          float = 0.9,    # nucleus sampling — cuts off tail of distribution
        repetition_penalty: float = 1.1, # > 1.0 penalizes repeated tokens
        stream:         bool  = True,   # print tokens as they generate
        device:         str   = None,   # None = auto detect
    ):
        self.max_new_tokens     = max_new_tokens
        self.temperature        = temperature
        self.top_k              = top_k
        self.top_p              = top_p
        self.repetition_penalty = repetition_penalty
        self.stream             = stream
        self.device             = device or ("cuda" if torch.cuda.is_available() else "cpu")


# ─── sampling ─────────────────────────────────────────────────────────────────

def _sample(logits: torch.Tensor, cfg: InferenceConfig, generated: torch.Tensor) -> int:
    logits = logits / max(cfg.temperature, 1e-8)

    # repetition penalty — downscale tokens already generated
    if cfg.repetition_penalty != 1.0:
        for token_id in generated.squeeze().tolist():
            logits[token_id] /= cfg.repetition_penalty

    # top-k
    if cfg.top_k > 0:
        top_k         = min(cfg.top_k, logits.size(-1))
        thresh        = logits.topk(top_k).values[..., -1, None]
        logits        = logits.masked_fill(logits < thresh, float("-inf"))

    # top-p (nucleus)
    if cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs                 = sorted_logits.softmax(-1).cumsum(-1)
        remove                    = cum_probs - sorted_logits.softmax(-1) > cfg.top_p
        sorted_logits[remove]     = float("-inf")
        logits                    = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

    return torch.multinomial(logits.softmax(-1), num_samples=1).item()


# ─── inference engine ─────────────────────────────────────────────────────────

class Inference:
    def __init__(self, model, tokenizer, cfg: InferenceConfig = None):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg or InferenceConfig()
        self.device    = torch.device(self.cfg.device)

        self.model.to(self.device)
        self.model.eval()

        _section("🧠 Inference Engine")
        _info("device",      self.cfg.device)
        _info("temperature", str(self.cfg.temperature))
        _info("top_k",       str(self.cfg.top_k))
        _info("top_p",       str(self.cfg.top_p))
        _info("streaming",   str(self.cfg.stream))

    @classmethod
    def from_checkpoint(cls, path: str, tokenizer, cfg: InferenceConfig = None):
        """Load model from a local checkpoint file."""
        from transformer_toolkit.model import Transformer, TransformerConfig

        _section("📂 Loading checkpoint")
        _info("path", path)

        ckpt   = torch.load(path, map_location="cpu", weights_only=False)
        config = ckpt.get("config") or TransformerConfig()
        model  = Transformer(config)
        model.load_state_dict(ckpt["model"])
        _ok(f"loaded  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        return cls(model, tokenizer, cfg)

    @classmethod
    def from_hub(cls, repo_id: str, tokenizer, cfg: InferenceConfig = None):
        """Pull model from HuggingFace Hub and load it."""
        from transformer_toolkit.hf_hub import pull_from_hub
        pull_from_hub(repo_id)
        return cls.from_checkpoint("checkpoints/model.pt", tokenizer, cfg)

    # ── core generate ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, prompt: str, cfg: InferenceConfig = None) -> str:
        cfg       = cfg or self.cfg
        ids       = self.tokenizer.encode(prompt)
        tokens    = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated = tokens.clone()

        if cfg.stream:
            print(f"\n{C.DIM}{'─' * 52}{C.RESET}")
            print(f"{C.CYAN}{prompt}{C.RESET}", end="", flush=True)

        output_ids = []

        for _ in range(cfg.max_new_tokens):
            logits    = self.model(tokens)[:, -1, :]        # [1, vocab]
            next_id   = _sample(logits.squeeze(0), cfg, generated)
            output_ids.append(next_id)

            next_tok  = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            tokens    = torch.cat([tokens, next_tok], dim=-1)
            generated = tokens.clone()

            # stop at eos
            if hasattr(self.tokenizer, "eos_id") and next_id == self.tokenizer.eos_id:
                break

            if cfg.stream:
                word = self.tokenizer.decode([next_id])
                print(f"{C.WHITE}{word}{C.RESET}", end="", flush=True)

        if cfg.stream:
            print(f"\n{C.DIM}{'─' * 52}{C.RESET}\n")

        return self.tokenizer.decode(output_ids)

    # ── batch generate ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_batch(self, prompts: list[str], cfg: InferenceConfig = None) -> list[str]:
        """Generate for multiple prompts. Non-streaming only."""
        cfg     = cfg or self.cfg
        results = []

        print(f"\n{C.YELLOW}⏳ generating {len(prompts)} prompts...{C.RESET}")
        for i, prompt in enumerate(prompts):
            print(f"\r  {C.DIM}{i+1}/{len(prompts)}{C.RESET}", end="", flush=True)
            non_stream_cfg       = InferenceConfig(**cfg.__dict__)
            non_stream_cfg.stream = False
            results.append(self.generate(prompt, non_stream_cfg))

        print(f"\r  {C.GREEN}✓ done{C.RESET}              ")
        return results