# sft_dataloader.py
"""
SFT data pipeline. Supports three schemas (auto-detected):
  - "prompt_response"  : {"prompt": "...", "response": "..."}
  - "messages"         : {"messages": [{"role": ..., "content": ...}]}
  - "instruction"      : {"instruction": "...", "output": "...", "input": ""}

Four data sources:
  - from_sft_json()    : local .json / .jsonl file
  - from_sft_hf()      : HuggingFace dataset
  - from_sft_strings() : list of dicts in memory
  - from_sft_files()   : list of .json / .jsonl paths

Each DataLoader yields (x, y, loss_mask).
loss_mask=1 ONLY on the FINAL assistant turn (content + closer + EOS).
All prior assistant turns are treated as context (loss_mask=0).
"""

import json
import torch
from pathlib import Path
from typing import Literal
from torch.utils.data import Dataset, DataLoader, IterableDataset

from .c_tokenizers import BaseTokenizer
from .chat_template import ChatTemplate
from .colors import C, _section, _info, _ok

Schema = Literal["prompt_response", "messages", "instruction", "auto"]


# ─── schema detection & normalisation ────────────────────────────────────────

def _detect_schema(sample: dict) -> Schema:
    keys = set(sample.keys())
    if keys & {"messages", "conversation", "conversations"}:
        return "messages"
    if "prompt" in keys and "response" in keys:
        return "prompt_response"
    if "instruction" in keys and keys & {"output", "response"}:
        return "instruction"
    if "input" in keys and "output" in keys:
        return "instruction"
    raise ValueError(f"Cannot detect schema from keys {keys}")


def _normalise(sample: dict, schema: Schema) -> list[dict]:
    """Return a list of {"role": ..., "content": ...} dicts."""
    if schema == "auto":
        schema = _detect_schema(sample)

    if schema == "messages":
        msgs = (
            sample.get("messages")
            or sample.get("conversation")
            or sample.get("conversations")
            or []
        )
        out = []
        for m in msgs:
            role    = m.get("role") or m.get("from", "user")
            content = m.get("content") or m.get("value", "")
            if role == "human":             role = "user"
            if role in ("gpt", "model"):    role = "assistant"
            out.append({"role": role, "content": content})
        return out

    if schema == "prompt_response":
        return [
            {"role": "user",      "content": sample.get("prompt",   "")},
            {"role": "assistant", "content": sample.get("response", "")},
        ]

    # instruction schema
    inp    = sample.get("input", "")
    prompt = (
        f"{sample.get('instruction', '')}\n\n{inp}".strip()
        if inp
        else sample.get("instruction", "")
    )
    return [
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": sample.get("output") or sample.get("response", "")},
    ]


# ─── turn-level truncation ────────────────────────────────────────────────────

def _truncate_turns(
    msgs: list[dict],
    tokenizer,
    template: ChatTemplate,
    seq_len: int,
    eos_token: str,
) -> list[dict]:
    """
    Drop the oldest turns (in pairs) until the sequence fits within seq_len.
    Always preserves at least one user + one assistant turn.
    """
    while len(msgs) > 2:
        text, _ = template.format_messages(msgs, eos_token=eos_token)
        if len(tokenizer.encode(text)) <= seq_len + 1:
            break
        msgs = msgs[2:]           # drop oldest user+assistant pair from the front
    return msgs


# ─── encode one sample ────────────────────────────────────────────────────────

def _encode(
    sample: dict,
    tokenizer,
    template: ChatTemplate,
    seq_len: int,
    schema: Schema,
    bos_id: int | None,
    eos_id: int | None,
    pad_id: int,
    trunc: str,
):
    """
    Returns (ids, tok_mask, full_text, spans) or None if the sample is unusable.

    tok_mask has 1 only for tokens that belong to the FINAL assistant span
    (response content + closing special token + EOS).  All prior assistant
    turns are supervised at 0 so the model treats them as context.
    """
    msgs      = _normalise(sample, schema)
    eos_token = tokenizer.decode([eos_id]) if eos_id is not None else ""

    if trunc == "turn":
        msgs = _truncate_turns(msgs, tokenizer, template, seq_len, eos_token)

    full_text, spans = template.format_messages(msgs, eos_token=eos_token)

    # ── KEY FIX: supervise ONLY the last assistant span ──────────────────────
    # `spans` is a list of (start_char, end_char) for every assistant turn.
    # Using all spans trains on historical assistant replies, which causes the
    # model to reproduce earlier turns instead of learning to continue the
    # conversation.  We mask only spans[-1] so prior turns are pure context.
    if not spans:
        return None

    last_start, last_end = spans[-1]
    char_mask = [0] * len(full_text)
    for i in range(last_start, min(last_end, len(full_text))):
        char_mask[i] = 1
    # ─────────────────────────────────────────────────────────────────────────

    # Tokenise — BOS is prepended as a token (not part of full_text)
    ids = tokenizer.encode(full_text)
    if bos_id is not None:
        ids = [bos_id] + ids

    # Align char-level mask → token-level mask via majority vote
    tok_mask  = [0] if bos_id is not None else []
    offset    = 0
    start_idx = 1 if bos_id is not None else 0

    for tid in ids[start_idx:]:
        surface = tokenizer.decode([tid])
        end     = offset + len(surface)
        chunk   = char_mask[offset : end]
        tok_mask.append(1 if sum(chunk) / max(len(chunk), 1) >= 0.5 else 0)
        offset = min(end, len(full_text))

    # Truncate to seq_len + 1 (the extra token is the next-step target for the
    # last position; __getitem__ splits into x[:-1] / y[1:])
    ids      = ids[      : seq_len + 1]
    tok_mask = tok_mask[ : seq_len + 1]

    # Skip samples where no response tokens survive truncation
    if sum(tok_mask) == 0 or len(ids) < 2:
        return None

    return ids, tok_mask, full_text, spans


# ─── in-memory dataset ────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        template: ChatTemplate,
        seq_len: int,
        schema: Schema,
        bos_id: int | None,
        eos_id: int | None,
        pad_id: int,
        trunc: str,
    ):
        self.seq_len = seq_len
        self.pad_id  = pad_id
        self.items: list = []
        skipped = 0

        for i, s in enumerate(samples):
            print(f"\r  {C.DIM}tokenizing{C.RESET}  {i+1}/{len(samples)}", end="", flush=True)
            result = _encode(s, tokenizer, template, seq_len, schema,
                             bos_id, eos_id, pad_id, trunc)
            if result is None:
                skipped += 1
            else:
                self.items.append(result)
        print()

        if skipped:
            print(f"  {C.YELLOW}⚠  skipped {skipped} samples (no response tokens after truncation){C.RESET}")
        print(f"  {C.GREEN}✓{C.RESET}  {len(self.items):,} usable samples")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        ids, mask, _, _ = self.items[idx]
        n = self.seq_len + 1

        # Pad to fixed length
        if len(ids) < n:
            pad  = n - len(ids)
            ids  = ids  + [self.pad_id] * pad
            mask = mask + [0]           * pad

        ids  = ids[:n]
        mask = mask[:n]

        t = torch.tensor(ids,  dtype=torch.long)
        m = torch.tensor(mask, dtype=torch.float)
        # x = input tokens, y = shifted targets, loss_mask aligned to y
        return t[:-1], t[1:], m[1:]


# ─── streaming dataset ────────────────────────────────────────────────────────

class StreamingSFTDataset(IterableDataset):
    def __init__(
        self,
        source,
        tokenizer,
        template: ChatTemplate,
        seq_len: int,
        schema: Schema,
        bos_id: int | None,
        eos_id: int | None,
        pad_id: int,
        trunc: str,
    ):
        self.source = source
        self._enc_args = (tokenizer, template, seq_len, schema,
                          bos_id, eos_id, pad_id, trunc)

    def __iter__(self):
        tok, tpl, seq_len, schema, bos_id, eos_id, pad_id, trunc = self._enc_args
        for s in self.source:
            result = _encode(s, tok, tpl, seq_len, schema,
                             bos_id, eos_id, pad_id, trunc)
            if result is None:
                continue

            ids, mask, _, _ = result
            n = seq_len + 1

            if len(ids) < n:
                pad  = n - len(ids)
                ids  = ids  + [pad_id] * pad
                mask = mask + [0]      * pad

            ids  = ids[:n]
            mask = mask[:n]

            t = torch.tensor(ids,  dtype=torch.long)
            m = torch.tensor(mask, dtype=torch.float)
            yield t[:-1], t[1:], m[1:]


# ─── config ───────────────────────────────────────────────────────────────────

class SFTDataConfig:
    def __init__(
        self,
        tokenizer            = None,
        seq_len:     int     = 512,
        batch_size:  int     = 8,
        shuffle:     bool    = True,
        num_workers: int     = 0,
        split:       float   = 0.9,
        streaming:   bool    = False,
        pin_memory:  bool    = True,
        schema:      Schema  = "auto",
        template             = "chatml",
        bos_id:      int     = None,
        eos_id:      int     = None,
        pad_id:      int     = None,
        truncation_strategy: str = "token",
        debug:       bool    = False,
        debug_n:     int     = 2,
    ):
        self.seq_len             = seq_len
        self.batch_size          = batch_size
        self.shuffle             = shuffle
        self.num_workers         = num_workers
        self.split               = split
        self.streaming           = streaming
        self.pin_memory          = pin_memory
        self.schema              = schema
        self.truncation_strategy = truncation_strategy
        self.debug               = debug
        self.debug_n             = debug_n
        self.template = (
            template if isinstance(template, ChatTemplate)
            else ChatTemplate(template)
        )

        if tokenizer is not None:
            self.bos_id = bos_id if bos_id is not None else getattr(tokenizer, "bos_id", None)
            self.eos_id = eos_id if eos_id is not None else getattr(tokenizer, "eos_id", None)
            self.pad_id = pad_id if pad_id is not None else getattr(tokenizer, "pad_id", 1)
            _validate_template_tokens(tokenizer, self.template)
        else:
            self.bos_id = bos_id
            self.eos_id = eos_id
            self.pad_id = pad_id if pad_id is not None else 1


def _validate_template_tokens(tokenizer, template: ChatTemplate) -> None:
    """Ensure every special token in the template is a single vocabulary entry."""
    fragmented = [t for t in template.special_tokens
                  if len(tokenizer.encode(t)) != 1]
    if fragmented:
        raise RuntimeError(
            f"Template tokens {fragmented} are fragmented in this vocabulary.\n"
            "Register them at tokenizer train time — vocab is frozen after pretraining."
        )


# ─── debug display ────────────────────────────────────────────────────────────

def _debug(train_dl: DataLoader, cfg: SFTDataConfig, tokenizer=None) -> None:
    w = 62
    print(f"\n{C.BOLD}{C.MAGENTA}{'─'*w}{C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}  SFT Debug samples{C.RESET}")
    print(
        f"{C.DIM}  seq_len={cfg.seq_len}  batch={cfg.batch_size}  "
        f"schema={cfg.schema}  template={cfg.template.preset}  "
        f"trunc={cfg.truncation_strategy}{C.RESET}"
    )
    print(
        f"{C.DIM}  legend: {C.CYAN}cyan=context (prompt + prior turns){C.RESET}  "
        f"{C.GREEN}{C.BOLD}green=final assistant turn + EOS{C.RESET}"
    )
    print(f"{C.BOLD}{C.MAGENTA}{'─'*w}{C.RESET}\n")

    x_b, y_b, m_b = next(iter(train_dl))
    n = min(cfg.debug_n, x_b.shape[0])

    for i in range(n):
        x    = x_b[i].tolist()
        y    = y_b[i].tolist()
        mask = m_b[i].tolist()

        n_resp = int(sum(mask))
        n_pad  = sum(1 for t, m in zip(y, mask) if t == cfg.pad_id and m == 0.0)
        pct    = 100.0 * n_resp / max(len(x), 1)

        print(f"  {C.BOLD}── sample {i+1} ──{C.RESET}")
        print(
            f"  tokens   total={len(x)}  "
            f"response={C.GREEN}{n_resp}{C.RESET}  "
            f"pad≈{n_pad}  resp%={pct:.1f}%"
        )

        # Compact mask bar
        bar = ""
        bkt = max(1, len(mask) // 50)
        for j in range(0, len(mask), bkt):
            chunk_sum = sum(mask[j : j + bkt])
            chunk_len = max(len(mask[j : j + bkt]), 1)
            f = chunk_sum / chunk_len
            bar += (
                f"{C.GREEN}█{C.RESET}"  if f >= 0.75 else
                f"{C.YELLOW}▒{C.RESET}" if f >= 0.25 else
                f"{C.DIM}░{C.RESET}"
            )
        print(f"  mask bar {bar}\n")

        # Formatted view using stored full_text + spans
        dataset = train_dl.dataset
        if hasattr(dataset, "items") and i < len(dataset.items):
            _, _, full_text, spans = dataset.items[i]

            # Only the last span is supervised — build char mask accordingly
            char_mask = [0] * len(full_text)
            if spans:
                s, e = spans[-1]
                for ci in range(s, min(e, len(full_text))):
                    char_mask[ci] = 1

            # Colorise output
            out = ""
            cur = None
            cs  = 0
            for ci, c in enumerate(char_mask):
                if c != cur:
                    if cur is not None:
                        seg  = full_text[cs:ci]
                        out += (
                            f"{C.GREEN}{C.BOLD}{seg}{C.RESET}" if cur == 1
                            else f"{C.CYAN}{seg}{C.RESET}"
                        )
                    cur = c
                    cs  = ci
            if cur is not None:
                seg  = full_text[cs:]
                out += (
                    f"{C.GREEN}{C.BOLD}{seg}{C.RESET}" if cur == 1
                    else f"{C.CYAN}{seg}{C.RESET}"
                )

            print(f"  {C.DIM}── formatted view ──{C.RESET}")
            for line in out[:480].split("\n"):
                if line:
                    print(f"  {line}")
            print()

            # Prompt / response blobs
            supervised_chars = set(
                ci
                for s, e in spans[-1:]          # only final span
                for ci in range(s, min(e, len(full_text)))
            )
            clip   = lambda s, n=120: s[:n] + f"{C.DIM}…{C.RESET}" if len(s) > n else s
            p_text = "".join(c for ci, c in enumerate(full_text) if ci not in supervised_chars)
            r_text = "".join(c for ci, c in enumerate(full_text) if ci in supervised_chars)
            print(f"  {C.DIM}context :{C.RESET} {C.CYAN}{clip(repr(p_text))}{C.RESET}")
            print(f"  {C.DIM}response:{C.RESET} {C.GREEN}{C.BOLD}{clip(repr(r_text))}{C.RESET}")

        # Sanity checks
        print()
        ok = True
        if n_resp == 0:
            print(f"  {C.RED}✗  zero response tokens — check schema/template{C.RESET}")
            ok = False
        else:
            print(f"  {C.GREEN}✓  mask: {n_resp} response tokens ({pct:.1f}%){C.RESET}")

        if x[1:] != y[:-1]:
            print(f"  {C.RED}✗  alignment broken{C.RESET}")
            ok = False
        else:
            print(f"  {C.GREEN}✓  alignment: y = x shifted by 1{C.RESET}")

        n_content = len(x) - n_pad
        if n_pad > 0 and n_pad / max(n_content, 1) > 3.0:
            suggested = max(16, int(n_content * 1.25 // 16) * 16)
            print(f"  {C.YELLOW}⚠  heavy padding — try seq_len={suggested}{C.RESET}")

        if ok:
            print(f"  {C.GREEN}✓  all checks passed{C.RESET}")
        print()

    print(f"{C.BOLD}{C.MAGENTA}{'─'*w}{C.RESET}\n")


# ─── internal builders ────────────────────────────────────────────────────────

def _build(
    samples: list[dict],
    tokenizer,
    cfg: SFTDataConfig,
) -> tuple[SFTDataset, SFTDataset]:
    split_at = int(len(samples) * cfg.split)
    kw = dict(
        tokenizer = tokenizer,
        template  = cfg.template,
        seq_len   = cfg.seq_len,
        schema    = cfg.schema,
        bos_id    = cfg.bos_id,
        eos_id    = cfg.eos_id,
        pad_id    = cfg.pad_id,
        trunc     = cfg.truncation_strategy,
    )
    print(f"  {C.DIM}tokenizing train...{C.RESET}")
    train = SFTDataset(samples[:split_at], **kw)
    print(f"  {C.DIM}tokenizing val...{C.RESET}")
    val   = SFTDataset(samples[split_at:], **kw)
    return train, val


def _loaders(
    train_ds,
    val_ds,
    cfg: SFTDataConfig,
    shuffle: bool | None = None,
    tokenizer=None,
) -> tuple[DataLoader, DataLoader]:
    do_shuffle = cfg.shuffle if shuffle is None else shuffle
    loader_kw  = dict(
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
        pin_memory  = cfg.pin_memory,
    )
    train_dl = DataLoader(train_ds, shuffle=do_shuffle, **loader_kw)
    val_dl   = DataLoader(val_ds,   shuffle=False,      **loader_kw)

    if isinstance(train_ds, SFTDataset):
        print(
            f"  {C.DIM}train{C.RESET}  {len(train_ds):>8,} samples  "
            f"{C.DIM}│{C.RESET}  {C.YELLOW}{len(train_dl):,}{C.RESET} batches"
        )
        print(
            f"  {C.DIM}val  {C.RESET}  {len(val_ds):>8,} samples  "
            f"{C.DIM}│{C.RESET}  {C.YELLOW}{len(val_dl):,}{C.RESET} batches\n"
        )

    if cfg.debug:
        _debug(train_dl, cfg, tokenizer)

    return train_dl, val_dl


# ─── public API ───────────────────────────────────────────────────────────────

def from_sft_strings(
    samples: list[dict],
    tokenizer,
    cfg: SFTDataConfig,
) -> tuple[DataLoader, DataLoader]:
    _section("SFT String dataset")
    _info("records",              str(len(samples)))
    _info("schema",               cfg.schema)
    _info("template",             cfg.template.preset)
    _info("truncation_strategy",  cfg.truncation_strategy)
    if cfg.schema == "auto" and samples:
        _info("detected schema", _detect_schema(samples[0]))
    train, val = _build(samples, tokenizer, cfg)
    return _loaders(train, val, cfg, tokenizer=tokenizer)


def from_sft_json(
    path: str,
    tokenizer,
    cfg: SFTDataConfig,
) -> tuple[DataLoader, DataLoader]:
    _section("SFT JSON dataset")
    _info("path",     path)
    _info("template", cfg.template.preset)

    p = Path(path)
    if p.suffix == ".jsonl":
        samples = [
            json.loads(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        samples = json.loads(p.read_text(encoding="utf-8"))

    _info("records", f"{len(samples):,}")
    if cfg.schema == "auto" and samples:
        _info("detected schema", _detect_schema(samples[0]))
    train, val = _build(samples, tokenizer, cfg)
    return _loaders(train, val, cfg, tokenizer=tokenizer)


def from_sft_files(
    paths: list[str],
    tokenizer,
    cfg: SFTDataConfig,
) -> tuple[DataLoader, DataLoader]:
    _section("SFT File dataset")
    for p in paths:
        _info("file", p)

    def _iter_files(fps):
        for fp in fps:
            p = Path(fp)
            if p.suffix == ".jsonl":
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        yield json.loads(line)
            else:
                yield from json.loads(p.read_text(encoding="utf-8"))

    if cfg.streaming:
        split_at = max(1, int(len(paths) * cfg.split))
        kw = dict(
            tokenizer = tokenizer,
            template  = cfg.template,
            seq_len   = cfg.seq_len,
            schema    = cfg.schema,
            bos_id    = cfg.bos_id,
            eos_id    = cfg.eos_id,
            pad_id    = cfg.pad_id,
            trunc     = cfg.truncation_strategy,
        )
        train_ds = StreamingSFTDataset(_iter_files(paths[:split_at]),                **kw)
        val_ds   = StreamingSFTDataset(_iter_files(paths[split_at:] or paths[-1:]),  **kw)
        return _loaders(train_ds, val_ds, cfg, shuffle=False, tokenizer=tokenizer)

    samples = list(_iter_files(paths))
    _info("total records", f"{len(samples):,}")
    if cfg.schema == "auto" and samples:
        _info("detected schema", _detect_schema(samples[0]))
    train, val = _build(samples, tokenizer, cfg)
    return _loaders(train, val, cfg, tokenizer=tokenizer)


def from_sft_hf(
    dataset_name: str,
    tokenizer,
    cfg: SFTDataConfig,
    split: str = "train",
) -> tuple[DataLoader, DataLoader]:
    _section("SFT HuggingFace dataset")
    _info("dataset",   dataset_name)
    _info("split",     split)
    _info("template",  cfg.template.preset)
    _info("streaming", str(cfg.streaming))

    from datasets import load_dataset

    if cfg.streaming:
        val_n  = max(1, cfg.batch_size * 20)
        kw     = dict(
            tokenizer = tokenizer,
            template  = cfg.template,
            seq_len   = cfg.seq_len,
            schema    = cfg.schema,
            bos_id    = cfg.bos_id,
            eos_id    = cfg.eos_id,
            pad_id    = cfg.pad_id,
            trunc     = cfg.truncation_strategy,
        )
        ds       = load_dataset(dataset_name, split=split, streaming=True)
        train_ds = StreamingSFTDataset(ds.skip(val_n), **kw)
        val_ds   = StreamingSFTDataset(ds.take(val_n),  **kw)
        loader_kw = dict(
            batch_size  = cfg.batch_size,
            num_workers = cfg.num_workers,
            pin_memory  = cfg.pin_memory,
        )
        _ok("streaming SFT dataloaders ready")
        return DataLoader(train_ds, **loader_kw), DataLoader(val_ds, **loader_kw)

    print(f"  {C.YELLOW}⏳ downloading...{C.RESET}", flush=True)
    samples = list(load_dataset(dataset_name, split=split, streaming=False))
    _ok(f"downloaded {len(samples):,} samples")
    if cfg.schema == "auto" and samples:
        _info("detected schema", _detect_schema(samples[0]))
    train, val = _build(samples, tokenizer, cfg)
    return _loaders(train, val, cfg, tokenizer=tokenizer)