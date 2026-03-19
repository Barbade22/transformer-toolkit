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
loss_mask=1 on assistant content + closer + EOS, 0 everywhere else.
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


# ─── schema helpers ───────────────────────────────────────────────────────────

def _detect_schema(s: dict) -> Schema:
    k = set(s.keys())
    if k & {"messages", "conversation", "conversations"}: return "messages"
    if "prompt"       in k and "response"  in k:          return "prompt_response"
    if "instruction"  in k and k & {"output", "response"}: return "instruction"
    if "input"        in k and "output"    in k:           return "instruction"
    raise ValueError(f"Cannot detect schema from keys {k}")


def _normalise(s: dict, schema: Schema) -> list[dict]:
    if schema == "auto":
        schema = _detect_schema(s)

    if schema == "messages":
        msgs = s.get("messages") or s.get("conversation") or s.get("conversations") or []
        out  = []
        for m in msgs:
            role    = m.get("role") or m.get("from", "user")
            content = m.get("content") or m.get("value", "")
            if role == "human":          role = "user"
            if role in ("gpt", "model"): role = "assistant"
            out.append({"role": role, "content": content})
        return out

    if schema == "prompt_response":
        return [{"role": "user",      "content": s.get("prompt",   "")},
                {"role": "assistant", "content": s.get("response", "")}]

    if schema == "instruction":
        inp    = s.get("input", "")
        prompt = f"{s.get('instruction','')}\n\n{inp}".strip() if inp \
                 else s.get("instruction", "")
        return [{"role": "user",      "content": prompt},
                {"role": "assistant", "content": s.get("output") or s.get("response", "")}]


# ─── truncation ───────────────────────────────────────────────────────────────

def _truncate_turns(msgs, tokenizer, template, seq_len, eos_token):
    while len(msgs) > 2:
        text, _ = template.format_messages(msgs, eos_token=eos_token)
        if len(tokenizer.encode(text)) <= seq_len + 1:
            break
        msgs = msgs[:-2]
    return msgs


# ─── encode one sample ────────────────────────────────────────────────────────

def _encode(sample, tokenizer, template, seq_len, schema, bos_id, eos_id, pad_id, trunc):
    msgs      = _normalise(sample, schema)
    eos_token = tokenizer.decode([eos_id]) if eos_id is not None else ""

    if trunc == "turn":
        msgs = _truncate_turns(msgs, tokenizer, template, seq_len, eos_token)

    full_text, spans = template.format_messages(msgs, eos_token=eos_token)

    # char-level mask
    char_mask = [0] * len(full_text)
    for s, e in spans:
        for i in range(s, min(e, len(full_text))):
            char_mask[i] = 1

    # tokenise — BOS prepended as token only, EOS already in full_text
    ids = tokenizer.encode(full_text)
    if bos_id is not None:
        ids = [bos_id] + ids

    # align char mask → token mask via majority vote
    tok_mask  = [0] if bos_id is not None else []
    offset    = 0
    start_idx = 1 if bos_id is not None else 0
    for tid in ids[start_idx:]:
        s_tok = tokenizer.decode([tid])
        end   = offset + len(s_tok)
        chunk = char_mask[offset:end]
        tok_mask.append(1 if sum(chunk) / max(len(chunk), 1) >= 0.5 else 0)
        offset = min(end, len(full_text))

    # truncate
    ids      = ids[:seq_len + 1]
    tok_mask = tok_mask[:seq_len + 1]

    if sum(tok_mask) == 0 or len(ids) < 2:
        return None

    return ids, tok_mask, full_text, spans


# ─── dataset ──────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, template, seq_len,
                 schema, bos_id, eos_id, pad_id, trunc):
        self.seq_len = seq_len
        self.pad_id  = pad_id
        self.items   = []
        skipped      = 0

        for i, s in enumerate(samples):
            print(f"\r  {C.DIM}tokenizing{C.RESET}  {i+1}/{len(samples)}", end="", flush=True)
            r = _encode(s, tokenizer, template, seq_len, schema, bos_id, eos_id, pad_id, trunc)
            if r is None:
                skipped += 1
            else:
                self.items.append(r)
        print()

        if skipped:
            print(f"  {C.YELLOW}⚠  skipped {skipped} samples{C.RESET}")
        print(f"  {C.GREEN}✓{C.RESET}  {len(self.items):,} usable samples")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ids, mask, _, _ = self.items[idx]
        n = self.seq_len + 1
        if len(ids) < n:
            pad  = n - len(ids)
            ids  = ids  + [self.pad_id] * pad
            mask = mask + [0]           * pad
        ids  = ids[:n]
        mask = mask[:n]
        t    = torch.tensor(ids,  dtype=torch.long)
        m    = torch.tensor(mask, dtype=torch.float)
        return t[:-1], t[1:], m[1:]


class StreamingSFTDataset(IterableDataset):
    def __init__(self, source, tokenizer, template, seq_len,
                 schema, bos_id, eos_id, pad_id, trunc):
        self.source = source
        self.args   = (tokenizer, template, seq_len, schema, bos_id, eos_id, pad_id, trunc)

    def __iter__(self):
        tok, tpl, seq_len, schema, bos_id, eos_id, pad_id, trunc = self.args
        for s in self.source:
            r = _encode(s, tok, tpl, seq_len, schema, bos_id, eos_id, pad_id, trunc)
            if r is None: continue
            ids, mask, _, _ = r
            n = seq_len + 1
            if len(ids) < n:
                pad  = n - len(ids)
                ids  = ids  + [pad_id] * pad
                mask = mask + [0]      * pad
            ids  = ids[:n]
            mask = mask[:n]
            t    = torch.tensor(ids,  dtype=torch.long)
            m    = torch.tensor(mask, dtype=torch.float)
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
        self.template = template if isinstance(template, ChatTemplate) \
                        else ChatTemplate(template)

        if tokenizer is not None:
            self.bos_id = bos_id if bos_id is not None else getattr(tokenizer, "bos_id", None)
            self.eos_id = eos_id if eos_id is not None else getattr(tokenizer, "eos_id", None)
            self.pad_id = pad_id if pad_id is not None else getattr(tokenizer, "pad_id", 1)
            _validate(tokenizer, self.template)
        else:
            self.bos_id = bos_id
            self.eos_id = eos_id
            self.pad_id = pad_id if pad_id is not None else 1


def _validate(tokenizer, template):
    bad = [t for t in template.special_tokens
           if len(tokenizer.encode(t)) != 1]
    if bad:
        raise RuntimeError(
            f"Template tokens {bad} are fragmented in this vocabulary.\n"
            f"Register them at tokenizer train time — vocab is frozen after pretraining."
        )


# ─── debug ────────────────────────────────────────────────────────────────────

def _debug(train_dl, cfg, tokenizer=None):
    w = 62
    print(f"\n{C.BOLD}{C.MAGENTA}{'─'*w}{C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}  SFT Debug samples{C.RESET}")
    print(f"{C.DIM}  seq_len={cfg.seq_len}  batch={cfg.batch_size}  "
          f"schema={cfg.schema}  template={cfg.template.preset}  "
          f"trunc={cfg.truncation_strategy}{C.RESET}")
    print(f"{C.DIM}  legend: {C.CYAN}cyan=prompt{C.RESET}  "
          f"{C.GREEN}{C.BOLD}green=response+closer+EOS{C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}{'─'*w}{C.RESET}\n")

    x_b, y_b, m_b = next(iter(train_dl))
    n = min(cfg.debug_n, x_b.shape[0])

    for i in range(n):
        x    = x_b[i].tolist()
        y    = y_b[i].tolist()
        mask = m_b[i].tolist()

        n_resp = int(sum(mask))
        n_pad  = sum(1 for t, m in zip(y, mask) if t == cfg.pad_id and m == 0.)
        pct    = 100. * n_resp / max(len(x), 1)

        print(f"  {C.BOLD}── sample {i+1} ──{C.RESET}")
        print(f"  tokens   total={len(x)}  "
              f"response={C.GREEN}{n_resp}{C.RESET}  "
              f"pad≈{n_pad}  resp%={pct:.1f}%")

        # mask bar
        bar = ""
        bkt = max(1, len(mask) // 50)
        for j in range(0, len(mask), bkt):
            f = sum(mask[j:j+bkt]) / max(len(mask[j:j+bkt]), 1)
            bar += f"{C.GREEN}█{C.RESET}" if f >= .75 \
                   else f"{C.YELLOW}▒{C.RESET}" if f >= .25 \
                   else f"{C.DIM}░{C.RESET}"
        print(f"  mask bar {bar}\n")

        # formatted view using stored full_text + spans
        if hasattr(train_dl.dataset, "items") and i < len(train_dl.dataset.items):
            _, _, full_text, spans = train_dl.dataset.items[i]

            # build char mask
            cm = [0] * len(full_text)
            for s, e in spans:
                for ci in range(s, min(e, len(full_text))): cm[ci] = 1

            # colorise
            out = ""; cur = None; cs = 0
            for ci, c in enumerate(cm):
                if c != cur:
                    if cur is not None:
                        seg  = full_text[cs:ci]
                        out += f"{C.GREEN}{C.BOLD}{seg}{C.RESET}" if cur == 1 \
                               else f"{C.CYAN}{seg}{C.RESET}"
                    cur = c; cs = ci
            if cur is not None:
                seg  = full_text[cs:]
                out += f"{C.GREEN}{C.BOLD}{seg}{C.RESET}" if cur == 1 \
                       else f"{C.CYAN}{seg}{C.RESET}"

            print(f"  {C.DIM}── formatted view ──{C.RESET}")
            for line in out[:480].split("\n"):
                if line: print(f"  {line}")
            print()

            # prompt / response blobs
            rc     = set(ci for s, e in spans
                         for ci in range(s, min(e, len(full_text))))
            clip   = lambda s, n=120: s[:n] + f"{C.DIM}…{C.RESET}" if len(s) > n else s
            p_text = "".join(c for ci, c in enumerate(full_text) if ci not in rc)
            r_text = "".join(c for ci, c in enumerate(full_text) if ci in rc)
            print(f"  {C.DIM}prompt  :{C.RESET} {C.CYAN}{clip(repr(p_text))}{C.RESET}")
            print(f"  {C.DIM}response:{C.RESET} {C.GREEN}{C.BOLD}{clip(repr(r_text))}{C.RESET}")

        # sanity checks
        print()
        ok = True
        if n_resp == 0:
            print(f"  {C.RED}✗  zero response tokens — check schema/template{C.RESET}")
            ok = False
        else:
            print(f"  {C.GREEN}✓  mask: {n_resp} response tokens ({pct:.1f}%){C.RESET}")
        if x[1:] != y[:-1]:
            print(f"  {C.RED}✗  alignment broken{C.RESET}"); ok = False
        else:
            print(f"  {C.GREEN}✓  alignment: y = x shifted by 1{C.RESET}")
        if n_pad / max(len(x) - n_pad, 1) > 3.0:
            suggested = max(16, int((len(x)-n_pad) * 1.25 // 16) * 16)
            print(f"  {C.YELLOW}⚠  heavy padding — try seq_len={suggested}{C.RESET}")
        if ok: print(f"  {C.GREEN}✓  all checks passed{C.RESET}")
        print()

    print(f"{C.BOLD}{C.MAGENTA}{'─'*w}{C.RESET}\n")


# ─── shared builder ───────────────────────────────────────────────────────────

def _build(samples, tokenizer, cfg):
    at = int(len(samples) * cfg.split)
    kw = dict(tokenizer=tokenizer, template=cfg.template, seq_len=cfg.seq_len,
              schema=cfg.schema, bos_id=cfg.bos_id, eos_id=cfg.eos_id,
              pad_id=cfg.pad_id, trunc=cfg.truncation_strategy)
    print(f"  {C.DIM}tokenizing train...{C.RESET}")
    tr = SFTDataset(samples[:at], **kw)
    print(f"  {C.DIM}tokenizing val...{C.RESET}")
    vl = SFTDataset(samples[at:], **kw)
    return tr, vl


def _loaders(tr, vl, cfg, shuffle=None, tokenizer=None):
    s  = cfg.shuffle if shuffle is None else shuffle
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
              pin_memory=cfg.pin_memory)
    tl = DataLoader(tr, shuffle=s,     **kw)
    vl = DataLoader(vl, shuffle=False, **kw)
    if isinstance(tr, SFTDataset):
        print(f"  {C.DIM}train{C.RESET}  {len(tr):>8,} samples  "
              f"{C.DIM}│{C.RESET}  {C.YELLOW}{len(tl):,}{C.RESET} batches")
        print(f"  {C.DIM}val  {C.RESET}  {len(vl.dataset):>8,} samples  "
              f"{C.DIM}│{C.RESET}  {C.YELLOW}{len(vl):,}{C.RESET} batches\n")
    if cfg.debug:
        _debug(tl, cfg, tokenizer)
    return tl, vl


# ─── public API ───────────────────────────────────────────────────────────────

def from_sft_strings(samples, tokenizer, cfg):
    _section("SFT String dataset")
    _info("records",  str(len(samples)))
    _info("schema",   cfg.schema)
    _info("template", cfg.template.preset)
    _info("truncation_strategy", cfg.truncation_strategy)
    if cfg.schema == "auto" and samples:
        _info("detected schema", _detect_schema(samples[0]))
    tr, vl = _build(samples, tokenizer, cfg)
    return _loaders(tr, vl, cfg, tokenizer=tokenizer)


def from_sft_json(path, tokenizer, cfg):
    _section("SFT JSON dataset")
    _info("path", path)
    _info("template", cfg.template.preset)
    p = Path(path)
    if p.suffix == ".jsonl":
        samples = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        samples = json.loads(p.read_text(encoding="utf-8"))
    _info("records", f"{len(samples):,}")
    if cfg.schema == "auto" and samples:
        _info("detected schema", _detect_schema(samples[0]))
    tr, vl = _build(samples, tokenizer, cfg)
    return _loaders(tr, vl, cfg, tokenizer=tokenizer)


def from_sft_files(paths, tokenizer, cfg):
    _section("SFT File dataset")
    for p in paths: _info("file", p)

    def _iter(fps):
        for fp in fps:
            p = Path(fp)
            if p.suffix == ".jsonl":
                for l in p.read_text(encoding="utf-8").splitlines():
                    if l.strip(): yield json.loads(l)
            else:
                for item in json.loads(p.read_text(encoding="utf-8")): yield item

    if cfg.streaming:
        at = max(1, int(len(paths) * cfg.split))
        kw = dict(tokenizer=tokenizer, template=cfg.template, seq_len=cfg.seq_len,
                  schema=cfg.schema, bos_id=cfg.bos_id, eos_id=cfg.eos_id,
                  pad_id=cfg.pad_id, trunc=cfg.truncation_strategy)
        tr = StreamingSFTDataset(_iter(paths[:at]),              **kw)
        vl = StreamingSFTDataset(_iter(paths[at:] or paths[-1:]), **kw)
        return _loaders(tr, vl, cfg, shuffle=False, tokenizer=tokenizer)

    samples = list(_iter(paths))
    _info("total records", f"{len(samples):,}")
    if cfg.schema == "auto" and samples:
        _info("detected schema", _detect_schema(samples[0]))
    tr, vl = _build(samples, tokenizer, cfg)
    return _loaders(tr, vl, cfg, tokenizer=tokenizer)


def from_sft_hf(dataset_name, tokenizer, cfg, split="train"):
    _section("SFT HuggingFace dataset")
    _info("dataset",   dataset_name)
    _info("split",     split)
    _info("template",  cfg.template.preset)
    _info("streaming", str(cfg.streaming))

    from datasets import load_dataset

    if cfg.streaming:
        val_n = max(1, cfg.batch_size * 20)
        kw    = dict(tokenizer=tokenizer, template=cfg.template, seq_len=cfg.seq_len,
                     schema=cfg.schema, bos_id=cfg.bos_id, eos_id=cfg.eos_id,
                     pad_id=cfg.pad_id, trunc=cfg.truncation_strategy)
        ds  = load_dataset(dataset_name, split=split, streaming=True)
        tr  = StreamingSFTDataset(ds.skip(val_n), **kw)
        vl  = StreamingSFTDataset(ds.take(val_n),  **kw)
        dkw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                   pin_memory=cfg.pin_memory)
        _ok("streaming SFT dataloaders ready")
        return DataLoader(tr, **dkw), DataLoader(vl, **dkw)

    print(f"  {C.YELLOW}⏳ downloading...{C.RESET}", flush=True)
    samples = list(load_dataset(dataset_name, split=split, streaming=False))
    _ok(f"downloaded {len(samples):,} samples")
    if cfg.schema == "auto" and samples:
        _info("detected schema", _detect_schema(samples[0]))
    tr, vl = _build(samples, tokenizer, cfg)
    return _loaders(tr, vl, cfg, tokenizer=tokenizer)