# sft_dataloader.py

"""
SFT (Supervised Fine-Tuning) data pipeline.

Supports three JSON schemas (auto-detected, or override with schema=):
  - "prompt_response"  : {"prompt": "...", "response": "..."}
  - "messages"         : {"messages": [{"role": "user/assistant", "content": "..."}]}
  - "instruction"      : {"instruction": "...", "output": "...", "input": ""}  (Alpaca)

Supports four data sources:
  - from_sft_json()    : local .json / .jsonl file
  - from_sft_hf()      : HuggingFace dataset (streaming or in-memory)
  - from_sft_strings() : list of dicts in memory
  - from_sft_files()   : list of .json / .jsonl file paths

Each DataLoader yields (x, y, loss_mask) — loss_mask is 0 on prompt tokens,
1 on response tokens + assistant_closer + EOS.

Template / tokenizer contract
──────────────────────────────
All special tokens used by any ChatTemplate preset MUST be registered in the
tokenizer's vocabulary at *pretrain* time (via RustBPETokenizer.train()).
The vocabulary is frozen after pretraining — no tokens can be added at SFT time.

Call tok.validate_template(cfg.template) once at SFT startup to assert this.

Loss mask rules per assistant turn
────────────────────────────────────
  <|im_start|>assistant\n   →  loss=0  (header, model sees as context)
  response content           →  loss=1
  <|im_end|>\n               →  loss=1  (closer, model must learn to emit)
  [EOS]                      →  loss=1  (sequence terminator)
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Literal, Optional
from torch.utils.data import Dataset, DataLoader, IterableDataset

from .c_tokenizers import BaseTokenizer
from .colors import C, _section, _info, _ok


# ─── schema type ──────────────────────────────────────────────────────────────

Schema = Literal["prompt_response", "messages", "instruction", "auto"]


# ─── chat templates ───────────────────────────────────────────────────────────

class ChatTemplate:
    """
    Formats a list of {"role": ..., "content": ...} messages into a single
    string, and returns the character spans where loss should be computed
    (assistant content + closer + EOS).

    Built-in presets: "chatml", "llama3", "gemma", "alpaca", "raw"

    Special token contract
    ──────────────────────
    Each preset declares which tokens must exist as *single* vocabulary entries
    via its "special_tokens" list. These must all be registered at tokenizer
    train time. Call tok.validate_template(template) at SFT startup to verify.

    Loss mask per assistant turn
    ─────────────────────────────
      assistant_header   →  loss=0  (prompt boundary, model sees as context)
      response content   →  loss=1
      assistant_closer   →  loss=1  (model must learn to emit this to end turn)
      eos_token          →  loss=1  (injected as string via format_messages())

    Usage:
        tpl = ChatTemplate("chatml")
        text, spans = tpl.format_messages(messages, eos_token="[EOS]")
        # spans = list of (start, end) char offsets, one per assistant turn
    """

    PRESETS = {
        "chatml": {
            # tokens: <|im_start|> role \n  content  <|im_end|> \n
            "system_fmt":       "<|im_start|>system\n{content}<|im_end|>\n",
            "user_fmt":         "<|im_start|>user\n{content}<|im_end|>\n",
            "assistant_fmt":    "<|im_start|>assistant\n{content}<|im_end|>\n",
            "assistant_header": "<|im_start|>assistant\n",   # loss=0
            "assistant_closer": "<|im_end|>\n",              # loss=1
            # must be single tokens in the vocabulary
            "special_tokens":   ["<|im_start|>", "<|im_end|>"],
        },
        "llama3": {
            "system_fmt":       "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
            "user_fmt":         "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant_fmt":    "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant_header": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "assistant_closer": "<|eot_id|>",
            "special_tokens":   ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        },
        "gemma": {
            "system_fmt":       "<start_of_turn>system\n{content}<end_of_turn>\n",
            "user_fmt":         "<start_of_turn>user\n{content}<end_of_turn>\n",
            "assistant_fmt":    "<start_of_turn>model\n{content}<end_of_turn>\n",
            "assistant_header": "<start_of_turn>model\n",
            "assistant_closer": "<end_of_turn>\n",
            "special_tokens":   ["<start_of_turn>", "<end_of_turn>"],
        },
        "alpaca": {
            # plain-text markers — no special tokens needed
            "system_fmt":       "### System:\n{content}\n\n",
            "user_fmt":         "### Instruction:\n{content}\n\n",
            "assistant_fmt":    "### Response:\n{content}\n\n",
            "assistant_header": "### Response:\n",
            "assistant_closer": "\n",
            "special_tokens":   [],
        },
        "raw": {
            "system_fmt":       "System: {content}\n",
            "user_fmt":         "User: {content}\n",
            "assistant_fmt":    "Assistant: {content}\n",
            "assistant_header": "Assistant: ",
            "assistant_closer": "\n",
            "special_tokens":   [],
        },
    }

    def __init__(
        self,
        preset:           str  = "chatml",
        system_fmt:       str  = None,
        user_fmt:         str  = None,
        assistant_fmt:    str  = None,
        assistant_header: str  = None,
        assistant_closer: str  = None,
        special_tokens:   list = None,
    ):
        """
        preset           : one of "chatml", "llama3", "gemma", "alpaca", "raw"
        system_fmt       : override system turn format  (must contain {content})
        user_fmt         : override user turn format
        assistant_fmt    : override assistant turn format
        assistant_header : prefix before response content — gets loss=0
        assistant_closer : suffix after response content — gets loss=1
                           (model must learn to emit this to close the turn)
        special_tokens   : list of strings that must be single vocab tokens
        """
        if preset not in self.PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. "
                f"Choose from {list(self.PRESETS)}"
            )
        base = self.PRESETS[preset]
        self.preset           = preset
        self.system_fmt       = system_fmt       or base["system_fmt"]
        self.user_fmt         = user_fmt         or base["user_fmt"]
        self.assistant_fmt    = assistant_fmt    or base["assistant_fmt"]
        self.assistant_header = assistant_header or base["assistant_header"]
        self.assistant_closer = assistant_closer or base["assistant_closer"]
        self.special_tokens   = special_tokens   if special_tokens is not None \
                                                 else list(base["special_tokens"])

    # ── formatting ────────────────────────────────────────────────────────────

    def format_messages(
        self,
        messages:  list[dict],
        eos_token: str = "",
    ) -> tuple[str, list[tuple[int, int]]]:
        """
        Format a multi-turn conversation into a single string.

        Returns:
            full_text       : complete formatted string ready for tokenisation
            response_spans  : list of (start, end) char offsets, one per
                              assistant turn. Each span covers:
                                  content + assistant_closer [+ eos_token]
                              i.e. everything the model must learn to generate.

        eos_token is appended as a *string* inside the span so that
        _align_mask assigns it loss=1 correctly. Pass tokenizer.decode([eos_id]).
        """
        text           = ""
        response_spans = []

        for msg in messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                text += self.system_fmt.format(content=content)

            elif role == "user":
                text += self.user_fmt.format(content=content)

            elif role == "assistant":
                # ── loss=0 region ─────────────────────────────────────
                header     = self.assistant_header
                # ── loss=1 region ─────────────────────────────────────
                closer     = self.assistant_closer
                eos        = eos_token

                span_start = len(text) + len(header)
                span_end   = span_start + len(content) + len(closer) + len(eos)

                text += header + content + closer + eos
                response_spans.append((span_start, span_end))

        return text, response_spans

    def format_single(self, prompt: str, response: str) -> tuple[str, int]:
        """
        Convenience wrapper for single-turn prompt/response pairs.
        Returns (full_text, response_char_start).
        Back-compat with any code that calls format_single directly.
        """
        msgs = [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text, spans = self.format_messages(msgs)
        return text, spans[0][0]


# ─── schema detection ─────────────────────────────────────────────────────────

def _detect_schema(sample: dict) -> Schema:
    """
    Infer schema from the keys present in a sample dict.
    Raises ValueError if no known schema is detected.
    """
    keys = set(sample.keys())

    if "messages" in keys or "conversation" in keys or "conversations" in keys:
        return "messages"

    if "prompt" in keys and "response" in keys:
        return "prompt_response"

    if "instruction" in keys and ("output" in keys or "response" in keys):
        return "instruction"

    if "input" in keys and "output" in keys:
        return "instruction"

    raise ValueError(
        f"Cannot auto-detect schema from keys {keys}. "
        f"Pass schema= explicitly: 'prompt_response', 'messages', or 'instruction'."
    )


def _normalise(sample: dict, schema: Schema) -> dict:
    """
    Normalise any supported schema into the canonical form:
        {"messages": [{"role": ..., "content": ...}, ...]}
    """
    if schema == "auto":
        schema = _detect_schema(sample)

    if schema == "messages":
        msgs = (sample.get("messages")
                or sample.get("conversation")
                or sample.get("conversations")
                or [])
        normalised = []
        for m in msgs:
            role    = m.get("role") or m.get("from", "user")
            content = m.get("content") or m.get("value", "")
            if role in ("human",):       role = "user"
            if role in ("gpt", "model"): role = "assistant"
            normalised.append({"role": role, "content": content})
        return {"messages": normalised}

    if schema == "prompt_response":
        return {"messages": [
            {"role": "user",      "content": sample.get("prompt",   "")},
            {"role": "assistant", "content": sample.get("response", "")},
        ]}

    if schema == "instruction":
        instruction = sample.get("instruction", "")
        inp         = sample.get("input", "")
        output      = sample.get("output") or sample.get("response", "")
        prompt      = f"{instruction}\n\n{inp}".strip() if inp else instruction
        return {"messages": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": output},
        ]}

    raise ValueError(f"Unknown schema: {schema!r}")


# ─── turn-aware truncation ────────────────────────────────────────────────────

def _truncate_to_turns(
    messages:  list[dict],
    tokenizer: BaseTokenizer,
    template:  ChatTemplate,
    seq_len:   int,
    eos_token: str = "",
) -> list[dict]:
    """
    Drop assistant+user turn pairs from the END of the conversation
    until the encoded length fits within seq_len + 1.

    Always keeps at least the first user + assistant pair so we never
    return an empty or unlearnable sample.

    Why: naive token truncation mid-assistant-turn leaves a partial response
    with loss=1 on incomplete/garbage text. Dropping whole turns is always safer.
    """
    msgs = list(messages)
    while len(msgs) > 2:
        text, _ = template.format_messages(msgs, eos_token=eos_token)
        ids     = tokenizer.encode(text)
        if len(ids) <= seq_len + 1:
            break
        # drop the last user+assistant pair (keep at least first pair)
        msgs = msgs[:-2]
    return msgs


# ─── tokenisation with loss mask ──────────────────────────────────────────────

def _encode_with_mask(
    sample:               dict,
    tokenizer:            BaseTokenizer,
    template:             ChatTemplate,
    seq_len:              int,
    schema:               Schema = "auto",
    bos_id:               int    = None,
    eos_id:               int    = None,
    pad_id:               int    = 1,
    truncation_strategy:  str    = "token",
) -> tuple[list[int], list[int]] | None:
    """
    Tokenise one sample and compute its loss mask.

    Returns (token_ids, loss_mask) where both are lists of length <= seq_len+1,
    or None if the sample produces 0 response tokens (nothing to learn from).

    loss_mask[i] = 1  →  token i is part of an assistant response /
                          closer / EOS  (compute loss here)
    loss_mask[i] = 0  →  prompt / system / user / header token (ignore)

    truncation_strategy:
        "token" — hard-truncate at seq_len (fine for single-turn SFT)
        "turn"  — drop whole turn pairs from the end until it fits
                  (correct for multi-turn conversation training)
    """
    norm = _normalise(sample, schema)
    msgs = norm["messages"]

    # decode EOS id → string so the template embeds it inside the char span
    # this ensures _align_mask assigns EOS loss=1 correctly
    eos_token = ""
    if eos_id is not None:
        try:
            eos_token = tokenizer.decode([eos_id])
        except Exception:
            eos_token = ""

    # ── truncation ────────────────────────────────────────────────────────────
    if truncation_strategy == "turn":
        msgs = _truncate_to_turns(msgs, tokenizer, template, seq_len, eos_token)

    # ── format and build char-level mask ──────────────────────────────────────
    full_text, response_spans = template.format_messages(msgs, eos_token=eos_token)

    char_mask = [0] * len(full_text)
    for start, end in response_spans:
        for i in range(start, min(end, len(full_text))):
            char_mask[i] = 1

    # ── tokenise ──────────────────────────────────────────────────────────────
    # BOS prepended as a token only — not part of the formatted text,
    # never included in any loss span.
    # EOS is already embedded inside full_text by format_messages(), so we do
    # NOT append it again here — that would double it.
    ids = tokenizer.encode(full_text)
    if bos_id is not None:
        ids = [bos_id] + ids

    # ── align char mask → token mask ──────────────────────────────────────────
    token_mask = _align_mask(full_text, char_mask, ids, tokenizer, bos_id)

    # ── hard token truncation (always applied as a safety cap) ────────────────
    max_len    = seq_len + 1
    ids        = ids[:max_len]
    token_mask = token_mask[:max_len]

    # skip samples with no response tokens — nothing to learn from
    if sum(token_mask) == 0:
        return None

    return ids, token_mask, full_text, response_spans


def _align_mask(
    text:      str,
    char_mask: list[int],
    token_ids: list[int],
    tokenizer: BaseTokenizer,
    bos_id:    int = None,
) -> list[int]:
    """
    Map a character-level binary mask to a token-level binary mask.

    Strategy: decode each token individually and use a running character cursor.
    Majority vote: if >= 50% of the characters a token covers are in a response
    span, the whole token gets loss=1.
    """
    token_mask = []
    offset     = 0

    start_idx = 1 if bos_id is not None else 0

    if bos_id is not None:
        token_mask.append(0)   # BOS is always masked out

    for tok_id in token_ids[start_idx:]:
        tok_str = tokenizer.decode([tok_id])
        end     = offset + len(tok_str)
        span    = char_mask[offset:end]
        vote    = sum(span) / max(len(span), 1)
        token_mask.append(1 if vote >= 0.5 else 0)
        offset  = min(end, len(text))

    return token_mask


# ─── dataset classes ──────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    In-memory SFT dataset. Pre-tokenises all samples at construction time.
    Yields (x, y, loss_mask) tensors of exactly seq_len tokens.
    """
    def __init__(
        self,
        samples:              list[dict],
        tokenizer:            BaseTokenizer,
        template:             ChatTemplate,
        seq_len:              int,
        schema:               Schema = "auto",
        bos_id:               int    = None,
        eos_id:               int    = None,
        pad_id:               int    = 1,
        truncation_strategy:  str    = "token",
    ):
        self.seq_len = seq_len
        self.pad_id  = pad_id
        self.items   = []

        skipped = 0
        for i, sample in enumerate(samples):
            print(f"\r  {C.DIM}tokenizing{C.RESET}  {i+1}/{len(samples)}", end="", flush=True)
            result = _encode_with_mask(
                sample, tokenizer, template, seq_len,
                schema, bos_id, eos_id, pad_id, truncation_strategy,
            )
            if result is None:
                skipped += 1
                continue
            ids, mask, full_text, response_spans = result
            if len(ids) < 2:
                skipped += 1
                continue
            self.items.append((ids, mask, full_text, response_spans))
        print()

        if skipped:
            print(f"  {C.YELLOW}⚠  skipped {skipped} samples "
                  f"(no response tokens or too short){C.RESET}")
        print(f"  {C.GREEN}✓{C.RESET}  {len(self.items):,} usable samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ids, mask, full_text, response_spans = self.items[idx]
        target_len = self.seq_len + 1

        if len(ids) < target_len:
            pad_len = target_len - len(ids)
            ids     = ids  + [self.pad_id] * pad_len   # pad with pad_id, not 0
            mask    = mask + [0]           * pad_len

        ids  = ids[:target_len]
        mask = mask[:target_len]

        ids_t  = torch.tensor(ids,  dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.float)

        x         = ids_t[:-1]
        y         = ids_t[1:]
        loss_mask = mask_t[1:]   # aligned with y (the targets)

        return x, y, loss_mask


class StreamingSFTDataset(IterableDataset):
    """
    Streaming SFT dataset — tokenises on the fly. No RAM limit.
    Yields (x, y, loss_mask) tensors.
    """
    def __init__(
        self,
        source,
        tokenizer:            BaseTokenizer,
        template:             ChatTemplate,
        seq_len:              int,
        schema:               Schema = "auto",
        bos_id:               int    = None,
        eos_id:               int    = None,
        pad_id:               int    = 1,
        truncation_strategy:  str    = "token",
    ):
        self.source               = source
        self.tokenizer            = tokenizer
        self.template             = template
        self.seq_len              = seq_len
        self.schema               = schema
        self.bos_id               = bos_id
        self.eos_id               = eos_id
        self.pad_id               = pad_id
        self.truncation_strategy  = truncation_strategy

    def __iter__(self):
        for sample in self.source:
            result = _encode_with_mask(
                sample, self.tokenizer, self.template,
                self.seq_len, self.schema,
                self.bos_id, self.eos_id, self.pad_id,
                self.truncation_strategy,
            )
            if result is None:
                continue
            ids, mask, full_text, response_spans = result
            if len(ids) < 2:
                continue

            target_len = self.seq_len + 1
            if len(ids) < target_len:
                pad_len = target_len - len(ids)
                ids  = ids  + [self.pad_id] * pad_len
                mask = mask + [0]           * pad_len

            ids  = ids[:target_len]
            mask = mask[:target_len]

            ids_t  = torch.tensor(ids,  dtype=torch.long)
            mask_t = torch.tensor(mask, dtype=torch.float)

            yield ids_t[:-1], ids_t[1:], mask_t[1:]


# ─── SFT DataConfig ───────────────────────────────────────────────────────────

class SFTDataConfig:
    """
    Configuration for SFT data loading.

    Pass tokenizer= to auto-pull bos_id / eos_id / pad_id.
    Override individually if needed.

    truncation_strategy:
        "token" — hard-truncate at seq_len (default, fine for single-turn)
        "turn"  — drop whole turn pairs from the end (correct for multi-turn
                  conversation training — never truncates mid-assistant-turn)
    """
    def __init__(
        self,
        tokenizer:            BaseTokenizer = None,
        seq_len:              int    = 512,
        batch_size:           int    = 8,
        shuffle:              bool   = True,
        num_workers:          int    = 0,
        split:                float  = 0.9,
        streaming:            bool   = False,
        pin_memory:           bool   = True,
        schema:               Schema = "auto",
        template:             str    = "chatml",
        bos_id:               int    = None,
        eos_id:               int    = None,
        pad_id:               int    = None,
        truncation_strategy:  str    = "token",
        debug:                bool   = False,
        debug_n:              int    = 2,
    ):
        # auto-pull from tokenizer; manual overrides take priority
        if tokenizer is not None:
            self.bos_id = bos_id if bos_id is not None else getattr(tokenizer, "bos_id", None)
            self.eos_id = eos_id if eos_id is not None else getattr(tokenizer, "eos_id", None)
            self.pad_id = pad_id if pad_id is not None else getattr(tokenizer, "pad_id", 1)
        else:
            self.bos_id = bos_id
            self.eos_id = eos_id
            self.pad_id = pad_id if pad_id is not None else 1

        self.seq_len              = seq_len
        self.batch_size           = batch_size
        self.shuffle              = shuffle
        self.num_workers          = num_workers
        self.split                = split
        self.streaming            = streaming
        self.pin_memory           = pin_memory
        self.schema               = schema
        self.truncation_strategy  = truncation_strategy
        self.debug                = debug
        self.debug_n              = debug_n

        # allow passing a pre-built ChatTemplate or just a preset name string
        self.template = (template if isinstance(template, ChatTemplate)
                         else ChatTemplate(template))

        # validate template tokens exist in the tokenizer vocabulary
        if tokenizer is not None:
            _validate_template_tokens(tokenizer, self.template)


def _validate_template_tokens(tokenizer: BaseTokenizer, template: ChatTemplate):
    """
    Assert that every special token the template uses encodes to exactly
    one token ID. Raises RuntimeError if any are missing or fragmented.

    This must be called at SFT startup — the vocabulary is frozen after
    pretraining and these tokens cannot be added retroactively.
    """
    missing = []
    for tok_str in template.special_tokens:
        try:
            ids = tokenizer.encode(tok_str)
            if len(ids) != 1:
                missing.append((tok_str, len(ids)))
        except Exception as e:
            missing.append((tok_str, str(e)))

    if missing:
        lines = "\n".join(f"  '{t}' → {n} tokens" for t, n in missing)
        raise RuntimeError(
            f"Template '{template.preset}' requires special tokens that are "
            f"fragmented or missing in this vocabulary:\n{lines}\n"
            f"All special tokens must be registered at tokenizer *pretrain* time.\n"
            f"Vocabulary is frozen — cannot add tokens at SFT time."
        )


# ─── debug helper ─────────────────────────────────────────────────────────────

def _render_masked_text(
    x_ids:     list[int],
    mask:      list[int],
    tokenizer: BaseTokenizer,
    max_chars: int = 480,
) -> str:
    """
    Decode the full token sequence and colorise by loss mask.
    cyan  = prompt / header tokens  (loss=0)
    green = response / closer / EOS (loss=1)

    Decodes full sequence at once — the only correct approach since
    the tokenizer decoder needs full context to restore spaces properly.
    Char boundaries are built by decoding increasing prefixes once,
    which is O(n) decode calls but done only in debug mode.
    """
    if not x_ids:
        return ""

    # decode full sequence at once for correct space restoration
    try:
        full_text = tokenizer.decode(x_ids)
    except Exception:
        full_text = "".join(f"[{t}]" for t in x_ids)

    if not full_text:
        return ""

    # build char boundaries via prefix decode
    # O(n) calls but only runs in debug mode — acceptable
    char_boundaries = [0]
    for k in range(1, len(x_ids) + 1):
        try:
            char_boundaries.append(len(tokenizer.decode(x_ids[:k])))
        except Exception:
            char_boundaries.append(char_boundaries[-1] + 1)

    # clamp boundaries to actual text length
    char_boundaries = [min(b, len(full_text)) for b in char_boundaries]

    # build char-level mask
    char_mask = [0] * len(full_text)
    for tok_idx, m in enumerate(mask):
        if tok_idx >= len(char_boundaries) - 1:
            break
        if m == 1:
            for ci in range(char_boundaries[tok_idx],
                            char_boundaries[tok_idx + 1]):
                char_mask[ci] = 1

    # group into contiguous segments and colorise
    segments  = []
    cur_mask  = None
    cur_start = 0
    for ci, cm in enumerate(char_mask):
        if cm != cur_mask:
            if cur_mask is not None:
                segments.append((cur_mask, full_text[cur_start:ci]))
            cur_mask  = cm
            cur_start = ci
    if cur_mask is not None:
        segments.append((cur_mask, full_text[cur_start:]))

    result    = ""
    total_len = 0
    truncated = False
    for seg_mask, seg_text in segments:
        if total_len >= max_chars:
            truncated = True
            break
        remaining = max_chars - total_len
        display   = seg_text[:remaining]
        if len(display) < len(seg_text):
            truncated = True
        if seg_mask == 1:
            result += f"{C.GREEN}{C.BOLD}{display}{C.RESET}"
        else:
            result += f"{C.CYAN}{display}{C.RESET}"
        total_len += len(display)

    if truncated:
        result += f"{C.DIM}…{C.RESET}"
    return result


def _mask_bar(mask: list[int], width: int = 50) -> str:
    n      = len(mask)
    bar    = ""
    bucket = max(1, n // width)
    for i in range(0, n, bucket):
        chunk = mask[i: i + bucket]
        frac  = sum(chunk) / max(len(chunk), 1)
        if frac >= 0.75:
            bar += f"{C.GREEN}█{C.RESET}"
        elif frac >= 0.25:
            bar += f"{C.YELLOW}▒{C.RESET}"
        else:
            bar += f"{C.DIM}░{C.RESET}"
    legend = (f"  {C.GREEN}█{C.RESET}=response  "
              f"{C.YELLOW}▒{C.RESET}=mixed  "
              f"{C.DIM}░{C.RESET}=prompt")
    return bar + legend


def _find_turn_boundaries(mask: list[int]) -> list[tuple[int, int, int]]:
    if not mask:
        return []
    runs  = []
    start = 0
    cur   = mask[0]
    for i in range(1, len(mask)):
        if mask[i] != cur:
            runs.append((start, i - 1, cur))
            start = i
            cur   = mask[i]
    runs.append((start, len(mask) - 1, cur))
    return runs


def _debug_sft_samples(
    train_dl:  DataLoader,
    cfg:       SFTDataConfig,
    tokenizer: BaseTokenizer = None,
):
    """
    Rich SFT debug output. For each sample shows:
      1. Token counts  — total / prompt / response / padding
      2. Mask bar      — compact visual of where loss is computed
      3. Turn breakdown
      4. Formatted template view (cyan=prompt, green=response+closer+EOS)
      5. Prompt / response decoded blobs
      6. Sanity checks
    """
    w = 62
    print(f"\n{C.BOLD}{C.MAGENTA}{'─' * w}{C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}  SFT Debug samples (train){C.RESET}")
    print(f"{C.DIM}  seq_len={cfg.seq_len}  batch={cfg.batch_size}  "
          f"schema={cfg.schema}  template={cfg.template.preset}  "
          f"trunc={cfg.truncation_strategy}{C.RESET}")
    print(f"{C.DIM}  legend:  "
          f"{C.CYAN}cyan = prompt/header (loss=0){C.RESET}  "
          f"{C.GREEN}{C.BOLD}green = response+closer+EOS (loss=1){C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}{'─' * w}{C.RESET}\n")

    batch = next(iter(train_dl))
    x_batch, y_batch, mask_batch = batch
    n = min(cfg.debug_n, x_batch.shape[0])

    for i in range(n):
        x_ids = x_batch[i].tolist()
        y_ids = y_batch[i].tolist()
        mask  = mask_batch[i].tolist()

        n_total    = len(x_ids)
        n_response = int(sum(mask))
        n_pad      = sum(1 for tok, m in zip(y_ids, mask)
                         if tok == cfg.pad_id and m == 0.0)
        resp_pct   = 100.0 * n_response / max(n_total, 1)

        print(f"  {C.BOLD}{C.WHITE}── sample {i+1} ──────────────────────────────────{C.RESET}")
        print(f"  {C.DIM}tokens   {C.RESET}  total={C.WHITE}{n_total}{C.RESET}"
              f"  response={C.GREEN}{C.BOLD}{n_response}{C.RESET}"
              f"  pad≈{C.DIM}{n_pad}{C.RESET}"
              f"  resp%={C.MAGENTA}{resp_pct:.1f}%{C.RESET}")

        print(f"  {C.DIM}mask bar {C.RESET}  {_mask_bar(mask)}")

        runs = _find_turn_boundaries(mask)
        turn_parts = []
        for start, end, mv in runs:
            length = end - start + 1
            label  = f"{C.GREEN}response{C.RESET}" if mv == 1 else f"{C.CYAN}prompt{C.RESET}"
            turn_parts.append(f"{label}[{start}:{end}]({length}tok)")
        print(f"  {C.DIM}turns    {C.RESET}  " + "  →  ".join(turn_parts))

        if tokenizer is not None:
            # ── formatted view — use stored full_text and response_spans ──────
            try:
                print(f"\n  {C.DIM}── formatted view ──{C.RESET}")
                # access dataset directly — no decode needed
                if hasattr(train_dl.dataset, 'items') and i < len(train_dl.dataset.items):
                    _, _, full_text, response_spans = train_dl.dataset.items[i]

                    # build char_mask from response_spans
                    char_mask = [0] * len(full_text)
                    for start, end in response_spans:
                        for ci in range(start, min(end, len(full_text))):
                            char_mask[ci] = 1

                    # colorise
                    result    = ""
                    cur_mask  = None
                    cur_start = 0
                    for ci, cm in enumerate(char_mask):
                        if cm != cur_mask:
                            if cur_mask is not None:
                                seg     = full_text[cur_start:ci]
                                result += f"{C.GREEN}{C.BOLD}{seg}{C.RESET}" \
                                          if cur_mask == 1 else f"{C.CYAN}{seg}{C.RESET}"
                            cur_mask  = cm
                            cur_start = ci
                    if cur_mask is not None:
                        seg     = full_text[cur_start:]
                        result += f"{C.GREEN}{C.BOLD}{seg}{C.RESET}" \
                                  if cur_mask == 1 else f"{C.CYAN}{seg}{C.RESET}"

                    for line in result[:480].split("\n"):
                        if line:
                            print(f"  {line}")
                    print()

                    # ── prompt / response blobs — slice full_text directly ────
                    def _clip(s, n=120):
                        return s[:n] + f"{C.DIM}…{C.RESET}" if len(s) > n else s

                    all_resp = set()
                    for start, end in response_spans:
                        for ci in range(start, min(end, len(full_text))):
                            all_resp.add(ci)

                    prompt_text   = "".join(c for ci, c in enumerate(full_text)
                                            if ci not in all_resp)
                    response_text = "".join(c for ci, c in enumerate(full_text)
                                            if ci in all_resp)

                    print(f"  {C.DIM}prompt  :{C.RESET} {C.CYAN}{_clip(repr(prompt_text))}{C.RESET}")
                    print(f"  {C.DIM}response:{C.RESET} {C.GREEN}{C.BOLD}{_clip(repr(response_text))}{C.RESET}")
                else:
                    # streaming dataset — fall back to decode
                    rendered = _render_masked_text(x_ids, mask, tokenizer, max_chars=480)
                    for line in rendered.split("\n"):
                        print(f"  {line}")
                    print()
            except Exception as e:
                print(f"  {C.YELLOW}⚠  render failed: {e}{C.RESET}\n")
        else:
            x_preview = x_ids[:16]
            tail      = f" … +{len(x_ids)-16}" if len(x_ids) > 16 else ""
            print(f"  {C.DIM}x ids    {C.RESET}  {C.CYAN}{x_preview}{tail}{C.RESET}")
            print(f"  {C.DIM}mask     {C.RESET}  {C.GREEN}{[int(m) for m in mask[:16]]}{tail}{C.RESET}")

        print()
        checks_ok = True

        if n_response == 0:
            print(f"  {C.RED}✗  MASK: zero response tokens — check schema / template{C.RESET}")
            checks_ok = False
        else:
            print(f"  {C.GREEN}✓  mask: {n_response} response tokens ({resp_pct:.1f}%){C.RESET}")

        if x_ids[1:] != y_ids[:-1]:
            print(f"  {C.RED}✗  ALIGNMENT: y is not x shifted by 1{C.RESET}")
            checks_ok = False
        else:
            print(f"  {C.GREEN}✓  alignment: y = x shifted by 1{C.RESET}")

        if len(set(x_ids)) == 1:
            print(f"  {C.RED}✗  TOKENS: all identical — possible tokenizer bug{C.RESET}")
            checks_ok = False

        if len(mask) != len(y_ids):
            print(f"  {C.RED}✗  LENGTH: mask {len(mask)} ≠ y {len(y_ids)}{C.RESET}")
            checks_ok = False

        n_content = n_total - n_pad
        if n_pad / max(n_content, 1) > 3.0:
            suggested = max(16, int(n_content * 1.25 // 16) * 16)
            print(f"  {C.YELLOW}⚠  PADDING: {n_pad} pad vs {n_content} content tokens "
                  f"— try seq_len={suggested}{C.RESET}")

        if n_response > 0 and resp_pct < 5.0:
            print(f"  {C.YELLOW}⚠  RATIO: only {resp_pct:.1f}% response tokens{C.RESET}")

        if checks_ok:
            print(f"  {C.GREEN}✓  all checks passed{C.RESET}")
        print()

    print(f"{C.BOLD}{C.MAGENTA}{'─' * w}{C.RESET}\n")


# ─── shared loader builder ────────────────────────────────────────────────────

def _build_datasets(
    samples:   list[dict],
    tokenizer: BaseTokenizer,
    cfg:       SFTDataConfig,
) -> tuple["SFTDataset", "SFTDataset"]:
    """Split samples and build train/val SFTDataset pairs."""
    split_at      = int(len(samples) * cfg.split)
    train_samples = samples[:split_at]
    val_samples   = samples[split_at:]

    print(f"  {C.DIM}tokenizing train...{C.RESET}")
    train_ds = SFTDataset(
        train_samples, tokenizer, cfg.template, cfg.seq_len,
        cfg.schema, cfg.bos_id, cfg.eos_id, cfg.pad_id, cfg.truncation_strategy,
    )
    print(f"  {C.DIM}tokenizing val...{C.RESET}")
    val_ds = SFTDataset(
        val_samples, tokenizer, cfg.template, cfg.seq_len,
        cfg.schema, cfg.bos_id, cfg.eos_id, cfg.pad_id, cfg.truncation_strategy,
    )
    return train_ds, val_ds


def _sft_loaders(
    train_ds,
    val_ds,
    cfg:       SFTDataConfig,
    shuffle:   bool            = None,
    tokenizer: BaseTokenizer   = None,
) -> tuple[DataLoader, DataLoader]:
    s        = cfg.shuffle if shuffle is None else shuffle
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=s,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    if isinstance(train_ds, Dataset):
        print(f"  {C.DIM}train{C.RESET}  {C.WHITE}{len(train_ds):>10,}{C.RESET} samples  "
              f"{C.DIM}│{C.RESET}  {C.YELLOW}{len(train_dl):,}{C.RESET} batches")
        print(f"  {C.DIM}val  {C.RESET}  {C.WHITE}{len(val_ds):>10,}{C.RESET} samples  "
              f"{C.DIM}│{C.RESET}  {C.YELLOW}{len(val_dl):,}{C.RESET} batches\n")

    if cfg.debug:
        _debug_sft_samples(train_dl, cfg, tokenizer)

    return train_dl, val_dl


# ─── public API ───────────────────────────────────────────────────────────────

def from_sft_json(
    path:      str,
    tokenizer: BaseTokenizer,
    cfg:       SFTDataConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Load from a local .json (list of dicts) or .jsonl (one dict per line) file.

    Schema is auto-detected from the first record, or override with cfg.schema.

    Example JSON:
        [{"prompt": "hi", "response": "hello"}, ...]

    Example JSONL (multi-turn):
        {"messages": [{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "hello"}]}
    """
    _section("SFT JSON dataset")
    _info("path",                 path)
    _info("schema",               cfg.schema)
    _info("template",             cfg.template.preset)
    _info("truncation_strategy",  cfg.truncation_strategy)

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

    train_ds, val_ds = _build_datasets(samples, tokenizer, cfg)
    return _sft_loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)


def from_sft_strings(
    samples:   list[dict],
    tokenizer: BaseTokenizer,
    cfg:       SFTDataConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Load from a list of dicts already in memory.
    Useful for quick experiments or programmatic dataset construction.
    """
    _section("SFT String dataset")
    _info("records",              str(len(samples)))
    _info("schema",               cfg.schema)
    _info("template",             cfg.template.preset)
    _info("truncation_strategy",  cfg.truncation_strategy)

    if cfg.schema == "auto" and samples:
        _info("detected schema", _detect_schema(samples[0]))

    train_ds, val_ds = _build_datasets(samples, tokenizer, cfg)
    return _sft_loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)


def from_sft_files(
    paths:     list[str],
    tokenizer: BaseTokenizer,
    cfg:       SFTDataConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Load and concatenate multiple .json / .jsonl files.
    If cfg.streaming=True, splits files into train/val groups and streams
    them without loading everything into RAM.
    """
    _section("SFT File dataset")
    for p in paths:
        _info("file", p)
    _info("schema",               cfg.schema)
    _info("template",             cfg.template.preset)
    _info("truncation_strategy",  cfg.truncation_strategy)

    if cfg.streaming:
        split_at    = max(1, int(len(paths) * cfg.split))
        train_paths = paths[:split_at]
        val_paths   = paths[split_at:] or paths[-1:]

        def _file_iter(file_paths):
            for fp in file_paths:
                p = Path(fp)
                if p.suffix == ".jsonl":
                    for line in p.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            yield json.loads(line)
                else:
                    for item in json.loads(p.read_text(encoding="utf-8")):
                        yield item

        train_ds = StreamingSFTDataset(
            _file_iter(train_paths), tokenizer, cfg.template, cfg.seq_len,
            cfg.schema, cfg.bos_id, cfg.eos_id, cfg.pad_id, cfg.truncation_strategy,
        )
        val_ds = StreamingSFTDataset(
            _file_iter(val_paths), tokenizer, cfg.template, cfg.seq_len,
            cfg.schema, cfg.bos_id, cfg.eos_id, cfg.pad_id, cfg.truncation_strategy,
        )
        return _sft_loaders(train_ds, val_ds, cfg, shuffle=False, tokenizer=tokenizer)

    # in-memory: load all files and concatenate
    all_samples = []
    for fp in paths:
        p = Path(fp)
        if p.suffix == ".jsonl":
            all_samples.extend(
                json.loads(line)
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        else:
            all_samples.extend(json.loads(p.read_text(encoding="utf-8")))

    _info("total records", f"{len(all_samples):,}")

    if cfg.schema == "auto" and all_samples:
        _info("detected schema", _detect_schema(all_samples[0]))

    train_ds, val_ds = _build_datasets(all_samples, tokenizer, cfg)
    return _sft_loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)


def from_sft_hf(
    dataset_name: str,
    tokenizer:    BaseTokenizer,
    cfg:          SFTDataConfig,
    split:        str = "train",
) -> tuple[DataLoader, DataLoader]:
    """
    Load from a HuggingFace dataset.
    Schema is auto-detected from the first row, or override with cfg.schema.
    Supports streaming (cfg.streaming=True) for large datasets.

    Example datasets:
        "OpenAssistant/oasst1"   → messages schema
        "tatsu-lab/alpaca"       → instruction schema
        "stanfordnlp/SHP"        → prompt_response schema
    """
    _section("SFT HuggingFace dataset")
    _info("dataset",              dataset_name)
    _info("split",                split)
    _info("schema",               cfg.schema)
    _info("template",             cfg.template.preset)
    _info("truncation_strategy",  cfg.truncation_strategy)
    _info("streaming",            str(cfg.streaming))

    from datasets import load_dataset

    if cfg.streaming:
        val_n    = max(1, cfg.batch_size * 20)
        ds_train = load_dataset(dataset_name, split=split, streaming=True)
        ds_val   = load_dataset(dataset_name, split=split, streaming=True)
        print(f"  {C.YELLOW}⚠  streaming: val = first {val_n} rows{C.RESET}")

        train_ds = StreamingSFTDataset(
            ds_train.skip(val_n), tokenizer, cfg.template, cfg.seq_len,
            cfg.schema, cfg.bos_id, cfg.eos_id, cfg.pad_id, cfg.truncation_strategy,
        )
        val_ds = StreamingSFTDataset(
            ds_val.take(val_n), tokenizer, cfg.template, cfg.seq_len,
            cfg.schema, cfg.bos_id, cfg.eos_id, cfg.pad_id, cfg.truncation_strategy,
        )
        train_dl = DataLoader(
            train_ds, batch_size=cfg.batch_size,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        )
        val_dl = DataLoader(
            val_ds, batch_size=cfg.batch_size,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        )
        _ok("streaming SFT dataloaders ready")
        return train_dl, val_dl

    # in-memory
    print(f"  {C.YELLOW}⏳ downloading...{C.RESET}", flush=True)
    ds      = load_dataset(dataset_name, split=split, streaming=False)
    samples = list(ds)
    _ok(f"downloaded {len(samples):,} samples")

    if cfg.schema == "auto" and samples:
        _info("detected schema", _detect_schema(samples[0]))

    train_ds, val_ds = _build_datasets(samples, tokenizer, cfg)
    return _sft_loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)