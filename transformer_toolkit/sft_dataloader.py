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
1 on response tokens. Pass all three to SFTTrainer.
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
    string, and returns the character offset where the first assistant response
    begins — so we can compute the loss mask precisely.

    Built-in templates: "chatml", "llama3", "alpaca", "raw"
    Custom: pass your own format strings.

    Usage:
        tpl = ChatTemplate("chatml")
        text, response_start = tpl.format([
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ])
    """

    PRESETS = {
        "chatml": {
            "system_fmt":    "<|im_start|>system\n{content}<|im_end|>\n",
            "user_fmt":      "<|im_start|>user\n{content}<|im_end|>\n",
            "assistant_fmt": "<|im_start|>assistant\n{content}<|im_end|>\n",
            "assistant_header": "<|im_start|>assistant\n",
        },
        "llama3": {
            "system_fmt":    "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
            "user_fmt":      "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant_fmt": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant_header": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        },
        "alpaca": {
            "system_fmt":    "### System:\n{content}\n\n",
            "user_fmt":      "### Instruction:\n{content}\n\n",
            "assistant_fmt": "### Response:\n{content}\n\n",
            "assistant_header": "### Response:\n",
        },
        "raw": {
            # No special tokens — just newline-separated turns.
            # Useful for simple models trained from scratch.
            "system_fmt":    "System: {content}\n",
            "user_fmt":      "User: {content}\n",
            "assistant_fmt": "Assistant: {content}\n",
            "assistant_header": "Assistant: ",
        },
    }

    def __init__(
        self,
        preset:           str  = "chatml",
        system_fmt:       str  = None,
        user_fmt:         str  = None,
        assistant_fmt:    str  = None,
        assistant_header: str  = None,
    ):
        """
        preset           : one of "chatml", "llama3", "alpaca", "raw"
        system_fmt       : override system turn format (must contain {content})
        user_fmt         : override user turn format
        assistant_fmt    : override assistant turn format
        assistant_header : the prefix string before the assistant response text
                           (used to locate where loss masking should end)
        """
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Choose from {list(self.PRESETS)}")
        base = self.PRESETS[preset]
        self.system_fmt       = system_fmt       or base["system_fmt"]
        self.user_fmt         = user_fmt         or base["user_fmt"]
        self.assistant_fmt    = assistant_fmt    or base["assistant_fmt"]
        self.assistant_header = assistant_header or base["assistant_header"]

    def format_single(self, prompt: str, response: str) -> tuple[str, int]:
        """
        Format a single prompt/response pair.
        Returns (full_text, response_char_start).
        """
        header = self.user_fmt.format(content=prompt)
        asst   = self.assistant_fmt.format(content=response)
        text   = header + self.assistant_header
        response_start = len(text)       # char offset where response tokens begin
        text  += response
        # close the assistant turn (everything after the response content)
        suffix = self.assistant_fmt.format(content=response)
        # suffix already contains the response; we just need the tail
        tail   = suffix[len(self.assistant_header) + len(response):]
        text  += tail
        return text, response_start

    def format_messages(self, messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
        """
        Format a multi-turn conversation.
        Returns:
            full_text           : the complete formatted string
            response_spans      : list of (start, end) char offsets for each
                                  assistant turn — only these positions get loss=1
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
                # record the span of the actual response text (not the header/footer)
                prefix = text + self.assistant_header
                start  = len(prefix)
                full   = self.assistant_fmt.format(content=content)
                tail   = full[len(self.assistant_header) + len(content):]
                end    = start + len(content)
                text   = prefix + content + tail
                response_spans.append((start, end))

        return text, response_spans


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

    # fallback hints
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
    Multi-turn messages pass through directly.
    Single-turn schemas are wrapped into a 2-message list.
    """
    if schema == "auto":
        schema = _detect_schema(sample)

    if schema == "messages":
        # support both "messages" and "conversations"/"conversation" keys
        msgs = (sample.get("messages")
                or sample.get("conversation")
                or sample.get("conversations")
                or [])
        # some datasets use "from"/"value" instead of "role"/"content"
        normalised = []
        for m in msgs:
            role    = m.get("role") or m.get("from", "user")
            content = m.get("content") or m.get("value", "")
            # map "human" → "user", "gpt"/"model" → "assistant" (common HF conventions)
            if role in ("human",):        role = "user"
            if role in ("gpt", "model"):  role = "assistant"
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


# ─── tokenisation with loss mask ──────────────────────────────────────────────

def _encode_with_mask(
    sample:    dict,
    tokenizer: BaseTokenizer,
    template:  ChatTemplate,
    seq_len:   int,
    schema:    Schema = "auto",
    bos_id:    int    = None,
    eos_id:    int    = None,
) -> tuple[list[int], list[int]] | None:
    """
    Tokenise one sample and compute its loss mask.

    Returns (token_ids, loss_mask) where both are lists of length <= seq_len+1,
    or None if the sample produces 0 response tokens (nothing to learn from).

    loss_mask[i] = 1  →  token i is part of an assistant response (compute loss)
    loss_mask[i] = 0  →  token i is a prompt / system / user token (ignore)
    """
    norm = _normalise(sample, schema)
    msgs = norm["messages"]

    full_text, response_spans = template.format_messages(msgs)

    # build a char-level mask first, then align to tokens
    char_mask = [0] * len(full_text)
    for start, end in response_spans:
        for i in range(start, min(end, len(full_text))):
            char_mask[i] = 1

    # tokenise the full text
    ids = tokenizer.encode(full_text)
    if bos_id is not None: ids = [bos_id] + ids
    if eos_id is not None: ids = ids + [eos_id]

    # align char mask to tokens via character-level byte offsets
    # We re-tokenise prefix strings of increasing length to find boundaries.
    # This is O(n²) but SFT datasets are small — acceptable.
    token_mask = _align_mask(full_text, char_mask, ids, tokenizer, bos_id)

    # truncate to seq_len + 1 (we need +1 for the x/y shift)
    max_len = seq_len + 1
    ids        = ids[:max_len]
    token_mask = token_mask[:max_len]

    # skip samples with no response tokens
    if sum(token_mask) == 0:
        return None

    return ids, token_mask


def _align_mask(
    text:      str,
    char_mask: list[int],
    token_ids: list[int],
    tokenizer: BaseTokenizer,
    bos_id:    int = None,
) -> list[int]:
    """
    Map a character-level binary mask to a token-level binary mask.

    Strategy: decode each token individually and use running character cursor.
    Falls back to majority-vote if decode lengths don't align exactly.
    """
    token_mask = []
    offset     = 0   # current char position in text

    start_idx = 1 if bos_id is not None else 0

    if bos_id is not None:
        token_mask.append(0)   # BOS is always masked out

    for tok_id in token_ids[start_idx:]:
        tok_str = tokenizer.decode([tok_id])
        end     = offset + len(tok_str)
        # majority vote: if more than half the characters this token covers
        # are in a response span, label the whole token as response
        span     = char_mask[offset:end]
        vote     = sum(span) / max(len(span), 1)
        token_mask.append(1 if vote >= 0.5 else 0)
        offset   = min(end, len(text))

    return token_mask


# ─── dataset classes ──────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    In-memory SFT dataset. Pre-tokenises all samples at construction time.
    Yields (x, y, loss_mask) tensors of exactly seq_len tokens.
    """
    def __init__(
        self,
        samples:   list[dict],
        tokenizer: BaseTokenizer,
        template:  ChatTemplate,
        seq_len:   int,
        schema:    Schema = "auto",
        bos_id:    int    = None,
        eos_id:    int    = None,
    ):
        self.seq_len = seq_len
        self.items   = []   # list of (ids, mask) pairs

        skipped = 0
        for i, sample in enumerate(samples):
            print(f"\r  {C.DIM}tokenizing{C.RESET}  {i+1}/{len(samples)}", end="", flush=True)
            result = _encode_with_mask(sample, tokenizer, template, seq_len, schema, bos_id, eos_id)
            if result is None:
                skipped += 1
                continue
            ids, mask = result
            if len(ids) < 2:
                skipped += 1
                continue
            self.items.append((ids, mask))
        print()

        if skipped:
            print(f"  {C.YELLOW}⚠  skipped {skipped} samples "
                  f"(no response tokens or too short){C.RESET}")
        print(f"  {C.GREEN}✓{C.RESET}  {len(self.items):,} usable samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ids, mask = self.items[idx]
        # pad or truncate to exactly seq_len + 1
        target_len = self.seq_len + 1
        if len(ids) < target_len:
            pad_len = target_len - len(ids)
            ids     = ids  + [0] * pad_len
            mask    = mask + [0] * pad_len

        ids  = ids[:target_len]
        mask = mask[:target_len]

        ids_t  = torch.tensor(ids,  dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.float)

        x          = ids_t[:-1]
        y          = ids_t[1:]
        loss_mask  = mask_t[1:]   # align mask with y (the targets)

        return x, y, loss_mask


class StreamingSFTDataset(IterableDataset):
    """
    Streaming SFT dataset — tokenises on the fly. No RAM limit.
    Yields (x, y, loss_mask) tensors.
    """
    def __init__(
        self,
        source,            # iterable of dicts
        tokenizer: BaseTokenizer,
        template:  ChatTemplate,
        seq_len:   int,
        schema:    Schema = "auto",
        bos_id:    int    = None,
        eos_id:    int    = None,
    ):
        self.source    = source
        self.tokenizer = tokenizer
        self.template  = template
        self.seq_len   = seq_len
        self.schema    = schema
        self.bos_id    = bos_id
        self.eos_id    = eos_id

    def __iter__(self):
        for sample in self.source:
            result = _encode_with_mask(
                sample, self.tokenizer, self.template,
                self.seq_len, self.schema, self.bos_id, self.eos_id
            )
            if result is None:
                continue
            ids, mask = result
            if len(ids) < 2:
                continue

            target_len = self.seq_len + 1
            if len(ids) < target_len:
                pad_len = target_len - len(ids)
                ids  = ids  + [0] * pad_len
                mask = mask + [0] * pad_len

            ids  = ids[:target_len]
            mask = mask[:target_len]

            ids_t  = torch.tensor(ids,  dtype=torch.long)
            mask_t = torch.tensor(mask, dtype=torch.float)

            yield ids_t[:-1], ids_t[1:], mask_t[1:]


# ─── SFT DataConfig ───────────────────────────────────────────────────────────

class SFTDataConfig:
    def __init__(
        self,
        seq_len:     int   = 512,
        batch_size:  int   = 8,
        shuffle:     bool  = True,
        num_workers: int   = 0,
        split:       float = 0.9,
        streaming:   bool  = False,
        pin_memory:  bool  = True,
        schema:      Schema = "auto",    # "auto" | "prompt_response" | "messages" | "instruction"
        template:    str   = "chatml",   # ChatTemplate preset or ChatTemplate instance
        bos_id:      int   = None,
        eos_id:      int   = None,
        debug:       bool  = False,
        debug_n:     int   = 2,
    ):
        self.seq_len     = seq_len
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.num_workers = num_workers
        self.split       = split
        self.streaming   = streaming
        self.pin_memory  = pin_memory
        self.schema      = schema
        self.bos_id      = bos_id
        self.eos_id      = eos_id
        self.debug       = debug
        self.debug_n     = debug_n
        # allow passing a pre-built ChatTemplate or just a preset name string
        self.template = (template if isinstance(template, ChatTemplate)
                         else ChatTemplate(template))


# ─── debug helper ─────────────────────────────────────────────────────────────

def _render_masked_text(
    x_ids:     list[int],
    mask:      list[int],
    tokenizer: BaseTokenizer,
    max_chars: int = 480,
) -> str:
    """
    Decode the full token sequence at once (so BPE joining works correctly),
    then walk character offsets to assign prompt/response colors.

    Prompt tokens   → cyan
    Response tokens → green + bold

    Strategy:
      1. Decode full sequence → full_text  (correct, no inter-token spacing)
      2. Decode prefix[0:i] for each boundary to find char offsets per token
      3. Build a char-level mask, then group into contiguous colored segments
    """
    if not x_ids:
        return ""

    # ── step 1: decode the full sequence properly ──────────────────────────
    try:
        full_text = tokenizer.decode(x_ids)
    except Exception:
        full_text = "".join(f"[{t}]" for t in x_ids)

    # ── step 2: find char boundary for each token via prefix decoding ──────
    # decode prefix of length k to get cumulative char offset at token k
    char_boundaries = [0]   # char_boundaries[k] = char offset after token k
    for k in range(1, len(x_ids) + 1):
        try:
            prefix_text = tokenizer.decode(x_ids[:k])
            char_boundaries.append(len(prefix_text))
        except Exception:
            # fallback: estimate boundary from previous
            char_boundaries.append(char_boundaries[-1] + 1)

    # ── step 3: build char-level mask ──────────────────────────────────────
    char_mask = [0] * len(full_text)
    for tok_idx, m in enumerate(mask):
        if tok_idx >= len(char_boundaries) - 1:
            break
        start = char_boundaries[tok_idx]
        end   = char_boundaries[tok_idx + 1]
        if m == 1:
            for ci in range(start, min(end, len(full_text))):
                char_mask[ci] = 1

    # ── step 4: group into contiguous segments and colorize ────────────────
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

    # ── step 5: build colored output, truncating at max_chars ──────────────
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
    """
    Compact visual of the loss mask as a bar.
    █ = response token (loss=1, green)
    ░ = prompt token   (loss=0, dim)
    Shows exact token positions at a glance.
    """
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
    """
    Find contiguous runs in the mask.
    Returns list of (start_idx, end_idx, mask_value) for each run.
    """
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


def _debug_sft_samples(train_dl: DataLoader, cfg: SFTDataConfig,
                       tokenizer: BaseTokenizer = None):
    """
    Rich SFT debug output. For each sample shows:

    1. Token counts  — total / prompt / response / padding
    2. Mask bar      — compact visual of where loss is computed
    3. Turn breakdown — each contiguous prompt/response run with token count
    4. Formatted template view — full decoded text with color coding
       cyan  = prompt tokens  (no loss)
       green = response tokens (loss computed here)
    5. Raw token ID preview (matches base Trainer style)
    6. Sanity checks — mask integrity, x/y alignment, all-identical token warning
    """
    w = 62
    print(f"\n{C.BOLD}{C.MAGENTA}{'─' * w}{C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}  🔍 SFT Debug samples (train){C.RESET}")
    print(f"{C.DIM}  seq_len={cfg.seq_len}  batch_size={cfg.batch_size}  "
          f"schema={cfg.schema}  template={type(cfg.template).__name__}{C.RESET}")
    print(f"{C.DIM}  legend:  "
          f"{C.CYAN}cyan = prompt (no loss){C.RESET}  "
          f"{C.GREEN}{C.BOLD}green = response (loss){C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}{'─' * w}{C.RESET}\n")

    batch = next(iter(train_dl))
    x_batch, y_batch, mask_batch = batch

    n = min(cfg.debug_n, x_batch.shape[0])

    for i in range(n):
        x_ids = x_batch[i].tolist()    # input tokens  (len = seq_len)
        y_ids = y_batch[i].tolist()    # target tokens (len = seq_len, = x shifted by 1)
        mask  = mask_batch[i].tolist() # loss mask aligned with y

        # ── token counts ──────────────────────────────────
        n_total    = len(x_ids)
        n_response = int(sum(mask))
        n_prompt   = sum(1 for m in mask if m == 0.0 and m == m)  # exclude NaN guard
        n_pad      = sum(1 for tok, m in zip(y_ids, mask) if tok == 0 and m == 0.0)
        resp_pct   = 100.0 * n_response / max(n_total, 1)

        print(f"  {C.BOLD}{C.WHITE}── sample {i+1} ──────────────────────────────────{C.RESET}")
        print(f"  {C.DIM}tokens   {C.RESET}  total={C.WHITE}{n_total}{C.RESET}"
              f"  prompt={C.CYAN}{n_prompt}{C.RESET}"
              f"  response={C.GREEN}{C.BOLD}{n_response}{C.RESET}"
              f"  pad≈{C.DIM}{n_pad}{C.RESET}"
              f"  resp%={C.MAGENTA}{resp_pct:.1f}%{C.RESET}")

        # ── mask bar ──────────────────────────────────────
        print(f"  {C.DIM}mask bar {C.RESET}  {_mask_bar(mask)}")

        # ── turn breakdown ────────────────────────────────
        runs = _find_turn_boundaries(mask)
        turn_parts = []
        for start, end, mv in runs:
            length = end - start + 1
            label  = f"{C.GREEN}response{C.RESET}" if mv == 1 else f"{C.CYAN}prompt{C.RESET}"
            turn_parts.append(f"{label}[{start}:{end}]({length}tok)")
        print(f"  {C.DIM}turns    {C.RESET}  " + "  →  ".join(turn_parts))

        # ── formatted template view ───────────────────────
        if tokenizer is not None:
            try:
                print(f"\n  {C.DIM}── formatted view (cyan=prompt / green=response) ──{C.RESET}")
                rendered = _render_masked_text(x_ids, mask, tokenizer, max_chars=480)
                # indent every line of the rendered text
                for line in rendered.split("\n"):
                    print(f"  {line}")
                print()
            except Exception as e:
                print(f"  {C.YELLOW}⚠  render failed: {e}{C.RESET}\n")

            # ── separate prompt / response decoded blobs ──
            try:
                # decode contiguous runs rather than scattered token IDs
                # so BPE merges are preserved within each segment
                runs    = _find_turn_boundaries(mask)
                p_parts = []
                r_parts = []
                for start, end, mv in runs:
                    seg_ids = x_ids[start: end + 1] if mv == 0 else y_ids[start: end + 1]
                    try:
                        decoded = tokenizer.decode(seg_ids)
                    except Exception:
                        decoded = " ".join(str(t) for t in seg_ids)
                    if mv == 0:
                        p_parts.append(decoded)
                    else:
                        r_parts.append(decoded)

                p_text = "".join(p_parts)
                r_text = "".join(r_parts)

                def _clip(s, n=120):
                    return s[:n] + f"{C.DIM}…{C.RESET}" if len(s) > n else s

                print(f"  {C.DIM}prompt  :{C.RESET} {C.CYAN}{_clip(repr(p_text))}{C.RESET}")
                print(f"  {C.DIM}response:{C.RESET} {C.GREEN}{C.BOLD}{_clip(repr(r_text))}{C.RESET}")
            except Exception as e:
                print(f"  {C.YELLOW}⚠  decode failed: {e}{C.RESET}")
        else:
            # no tokenizer — show raw IDs like base _debug_samples
            x_preview = x_ids[:16]
            tail      = f" … +{len(x_ids)-16}" if len(x_ids) > 16 else ""
            print(f"  {C.DIM}x ids    {C.RESET}  {C.CYAN}{x_preview}{tail}{C.RESET}")
            m_preview = [int(m) for m in mask[:16]]
            print(f"  {C.DIM}mask     {C.RESET}  {C.GREEN}{m_preview}{tail}{C.RESET}")

        # ── sanity checks ─────────────────────────────────
        print()
        checks_ok = True

        # 1. mask integrity
        if n_response == 0:
            print(f"  {C.RED}✗  MASK: zero response tokens — check schema / template{C.RESET}")
            checks_ok = False
        else:
            print(f"  {C.GREEN}✓  mask: {n_response} response tokens ({resp_pct:.1f}% of seq){C.RESET}")

        # 2. x/y alignment (y should be x shifted by 1)
        if x_ids[1:] != y_ids[:-1]:
            print(f"  {C.RED}✗  ALIGNMENT: y is not x shifted by 1 — check dataset{C.RESET}")
            checks_ok = False
        else:
            print(f"  {C.GREEN}✓  alignment: y = x shifted by 1{C.RESET}")

        # 3. all-identical token warning (tokenizer bug detector)
        if len(set(x_ids)) == 1:
            print(f"  {C.RED}✗  TOKENS: all input tokens identical — possible tokenizer bug{C.RESET}")
            checks_ok = False

        # 4. mask/y length match
        if len(mask) != len(y_ids):
            print(f"  {C.RED}✗  LENGTH: mask length {len(mask)} ≠ y length {len(y_ids)}{C.RESET}")
            checks_ok = False

        # 5. padding warning — warn only when padding dominates actual content
        n_content    = n_total - n_pad           # real tokens (prompt + response)
        pad_of_content = n_pad / max(n_content, 1)
        if pad_of_content > 3.0:                 # padding is >3x the real content
            suggested = max(16, int(n_content * 1.25 // 16) * 16)  # round up to nearest 16
            print(f"  {C.YELLOW}⚠  PADDING: {n_pad} pad tokens vs {n_content} content tokens "
                  f"— seq_len={cfg.seq_len} may be too large; "
                  f"try seq_len={suggested}{C.RESET}")

        # 6. response fraction sanity
        if n_response > 0 and resp_pct < 5.0:
            print(f"  {C.YELLOW}⚠  RATIO: only {resp_pct:.1f}% response tokens "
                  f"— prompt may be very long relative to response{C.RESET}")

        if checks_ok:
            print(f"  {C.GREEN}✓  all checks passed{C.RESET}")

        print()

    print(f"{C.BOLD}{C.MAGENTA}{'─' * w}{C.RESET}\n")


# ─── shared loader builder ────────────────────────────────────────────────────

def _sft_loaders(
    train_ds, val_ds, cfg: SFTDataConfig,
    shuffle=None, tokenizer: BaseTokenizer = None,
) -> tuple[DataLoader, DataLoader]:
    s        = cfg.shuffle if shuffle is None else shuffle
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=s,
                          num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    if isinstance(train_ds, Dataset):   # not streaming
        print(f"  {C.DIM}train{C.RESET}  {C.WHITE}{len(train_ds):>10,}{C.RESET} samples  "
              f"{C.DIM}│{C.RESET}  {C.YELLOW}{len(train_dl):,}{C.RESET} batches")
        print(f"  {C.DIM}val  {C.RESET}  {C.WHITE}{len(val_ds):>10,}{C.RESET} samples  "
              f"{C.DIM}│{C.RESET}  {C.YELLOW}{len(val_dl):,}{C.RESET} batches\n")

    if cfg.debug:
        _debug_sft_samples(train_dl, cfg, tokenizer)

    return train_dl, val_dl


def _split_samples(samples: list[dict], cfg: SFTDataConfig):
    split_at = int(len(samples) * cfg.split)
    return samples[:split_at], samples[split_at:]


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

    Example JSONL:
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        ...
    """
    _section("📄 SFT JSON dataset")
    _info("path",   path)
    _info("schema", cfg.schema)

    p = Path(path)
    if p.suffix == ".jsonl":
        samples = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        samples = json.loads(p.read_text(encoding="utf-8"))

    _info("records", f"{len(samples):,}")

    if cfg.schema == "auto" and samples:
        detected = _detect_schema(samples[0])
        _info("detected schema", detected)

    train_samples, val_samples = _split_samples(samples, cfg)

    print(f"  {C.DIM}tokenizing train...{C.RESET}")
    train_ds = SFTDataset(train_samples, tokenizer, cfg.template,
                          cfg.seq_len, cfg.schema, cfg.bos_id, cfg.eos_id)
    print(f"  {C.DIM}tokenizing val...{C.RESET}")
    val_ds   = SFTDataset(val_samples,   tokenizer, cfg.template,
                          cfg.seq_len, cfg.schema, cfg.bos_id, cfg.eos_id)

    return _sft_loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)


def from_sft_strings(
    samples:   list[dict],
    tokenizer: BaseTokenizer,
    cfg:       SFTDataConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Load from a list of dicts already in memory.

    Useful for quick experiments or programmatic dataset construction.
    Schema auto-detected unless cfg.schema is set.
    """
    _section("📝 SFT String dataset")
    _info("records", str(len(samples)))
    _info("schema",  cfg.schema)

    if cfg.schema == "auto" and samples:
        detected = _detect_schema(samples[0])
        _info("detected schema", detected)

    train_samples, val_samples = _split_samples(samples, cfg)

    print(f"  {C.DIM}tokenizing train...{C.RESET}")
    train_ds = SFTDataset(train_samples, tokenizer, cfg.template,
                          cfg.seq_len, cfg.schema, cfg.bos_id, cfg.eos_id)
    print(f"  {C.DIM}tokenizing val...{C.RESET}")
    val_ds   = SFTDataset(val_samples,   tokenizer, cfg.template,
                          cfg.seq_len, cfg.schema, cfg.bos_id, cfg.eos_id)

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
    _section("📂 SFT File dataset")
    for p in paths: _info("file", p)
    _info("schema", cfg.schema)

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

        train_ds = StreamingSFTDataset(_file_iter(train_paths), tokenizer,
                                       cfg.template, cfg.seq_len, cfg.schema,
                                       cfg.bos_id, cfg.eos_id)
        val_ds   = StreamingSFTDataset(_file_iter(val_paths),   tokenizer,
                                       cfg.template, cfg.seq_len, cfg.schema,
                                       cfg.bos_id, cfg.eos_id)
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
        detected = _detect_schema(all_samples[0])
        _info("detected schema", detected)

    train_samples, val_samples = _split_samples(all_samples, cfg)

    print(f"  {C.DIM}tokenizing train...{C.RESET}")
    train_ds = SFTDataset(train_samples, tokenizer, cfg.template,
                          cfg.seq_len, cfg.schema, cfg.bos_id, cfg.eos_id)
    print(f"  {C.DIM}tokenizing val...{C.RESET}")
    val_ds   = SFTDataset(val_samples,   tokenizer, cfg.template,
                          cfg.seq_len, cfg.schema, cfg.bos_id, cfg.eos_id)

    return _sft_loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)


def from_sft_hf(
    dataset_name: str,
    tokenizer:    BaseTokenizer,
    cfg:          SFTDataConfig,
    split:        str = "train",
    text_col:     str = None,       # None = auto-detect from schema
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
    _section("🤗 SFT HuggingFace dataset")
    _info("dataset",   dataset_name)
    _info("split",     split)
    _info("schema",    cfg.schema)
    _info("streaming", str(cfg.streaming))

    from datasets import load_dataset

    if cfg.streaming:
        val_n   = max(1, cfg.batch_size * 20)
        ds_train = load_dataset(dataset_name, split=split, streaming=True)
        ds_val   = load_dataset(dataset_name, split=split, streaming=True)

        print(f"  {C.YELLOW}⚠  streaming: val = first {val_n} rows{C.RESET}")

        train_ds = StreamingSFTDataset(
            ds_train.skip(val_n), tokenizer, cfg.template,
            cfg.seq_len, cfg.schema, cfg.bos_id, cfg.eos_id
        )
        val_ds = StreamingSFTDataset(
            ds_val.take(val_n), tokenizer, cfg.template,
            cfg.seq_len, cfg.schema, cfg.bos_id, cfg.eos_id
        )
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        _ok("streaming SFT dataloaders ready")
        return train_dl, val_dl

    # in-memory
    print(f"  {C.YELLOW}⏳ downloading...{C.RESET}", flush=True)
    ds      = load_dataset(dataset_name, split=split, streaming=False)
    samples = list(ds)
    _ok(f"downloaded {len(samples):,} samples")

    if cfg.schema == "auto" and samples:
        detected = _detect_schema(samples[0])
        _info("detected schema", detected)

    train_samples, val_samples = _split_samples(samples, cfg)

    print(f"  {C.DIM}tokenizing train...{C.RESET}")
    train_ds = SFTDataset(train_samples, tokenizer, cfg.template,
                          cfg.seq_len, cfg.schema, cfg.bos_id, cfg.eos_id)
    print(f"  {C.DIM}tokenizing val...{C.RESET}")
    val_ds   = SFTDataset(val_samples,   tokenizer, cfg.template,
                          cfg.seq_len, cfg.schema, cfg.bos_id, cfg.eos_id)

    return _sft_loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)