# dataloader.py

import array
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, IterableDataset
from .c_tokenizers import BaseTokenizer
from .colors import C, _section, _info, _ok, _bar


# ─── config ───────────────────────────────────────────────────────────────────

class DataConfig:
    def __init__(
        self,
        seq_len:     int   = 512,
        batch_size:  int   = 16,
        shuffle:     bool  = True,
        num_workers: int   = 0,
        split:       float = 0.9,
        streaming:   bool  = False,
        stride:      int   = None,
        pin_memory:  bool  = True,
        debug:       bool  = False,
        debug_n:     int   = 3,
    ):
        self.seq_len     = seq_len
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.num_workers = num_workers
        self.split       = split
        self.streaming   = streaming
        self.stride      = stride
        self.pin_memory  = pin_memory
        self.debug       = debug
        self.debug_n     = debug_n


# ─── core dataset (memmap) ────────────────────────────────────────────────────

class MemmapTokenDataset(Dataset):
    def __init__(self, token_path: str, seq_len: int, stride: int = None):
        self.seq_len   = seq_len
        self.stride    = stride or seq_len
        self.tokens    = np.load(token_path, mmap_mode='r')
        self.n_samples = max(0, (len(self.tokens) - self.seq_len - 1) // self.stride)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.tokens[start: start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


# ─── streaming dataset ────────────────────────────────────────────────────────

class StreamingDataset(IterableDataset):
    def __init__(self, paths: list[str], tokenizer: BaseTokenizer, seq_len: int):
        self.paths     = paths
        self.tokenizer = tokenizer
        self.seq_len   = seq_len

    def __iter__(self):
        buf = []
        for path in self.paths:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    buf.extend(self.tokenizer.encode(line))
                    while len(buf) >= self.seq_len + 1:
                        chunk = buf[:self.seq_len + 1]
                        buf   = buf[self.seq_len:]
                        yield (
                            torch.tensor(chunk[:-1], dtype=torch.long),
                            torch.tensor(chunk[1:],  dtype=torch.long),
                        )


# ─── HF streaming dataset ─────────────────────────────────────────────────────

class HFStreamDataset(IterableDataset):
    """
    Wraps a HuggingFace IterableDataset for token-chunked streaming.

    Tracks rows_seen (exact) for reliable resume via .skip().
    chunks_seen and tokens_seen are informational counters.

    state_dict() / load_state_dict() integrate with trainer checkpointing.
    """
    def __init__(
        self,
        dataset_name: str,
        split:        str,
        tokenizer:    BaseTokenizer,
        seq_len:      int,
        text_col:     str = "text",
        bos_id:       int = None,
        eos_id:       int = None,
        skip_rows:    int = 0,          # rows already consumed (restored on resume)
        take_rows:    int = None,       # None = stream forever; int = take N rows
        config:       str = None,       # HF dataset config e.g. 'sample-10BT'
    ):
        self.dataset_name = dataset_name
        self.split        = split
        self.config       = config
        self.tokenizer    = tokenizer
        self.seq_len      = seq_len
        self.text_col     = text_col
        self.bos_id       = bos_id
        self.eos_id       = eos_id
        self.skip_rows    = skip_rows
        self.take_rows    = take_rows

        # counters — mutated during iteration
        self.rows_seen   = skip_rows   # total rows consumed from source (survives resume)
        self.chunks_seen = 0           # chunks yielded this session
        self.tokens_seen = 0           # tokens yielded this session

    def _build_source(self):
        from datasets import load_dataset
        kwargs = dict(split=self.split, streaming=True)
        if self.config: kwargs['name'] = self.config
        ds = load_dataset(self.dataset_name, **kwargs)
        if self.skip_rows:
            ds = ds.skip(self.skip_rows)
        if self.take_rows is not None:
            ds = ds.take(self.take_rows)
        return ds

    def __iter__(self):
        buf = []
        for row in self._build_source():
            self.rows_seen += 1
            doc = self.tokenizer.encode(row[self.text_col])
            if self.bos_id is not None: doc = [self.bos_id] + doc
            if self.eos_id is not None: doc = doc + [self.eos_id]
            buf.extend(doc)
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf   = buf[self.seq_len:]      # advance by seq_len, not seq_len+1
                self.chunks_seen += 1
                self.tokens_seen += self.seq_len
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:],  dtype=torch.long),
                )

    def state_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "split":        self.split,
            "rows_seen":    self.rows_seen,
            "chunks_seen":  self.chunks_seen,
            "tokens_seen":  self.tokens_seen,
        }

    def load_state_dict(self, sd: dict):
        """Call before iterating to restore resume position."""
        self.rows_seen   = sd["rows_seen"]
        self.chunks_seen = sd.get("chunks_seen", 0)
        self.tokens_seen = sd.get("tokens_seen", 0)
        self.skip_rows   = sd["rows_seen"]   # rewind source to this row


# ─── binary / npy helpers ─────────────────────────────────────────────────────

def save_binary(tokens: list[int], path: str):
    max_id = max(tokens) if tokens else 0
    assert max_id < 65536, (
        f"Token ID {max_id} exceeds uint16 range (65535). "
        "Use a smaller vocabulary or switch to int32 storage."
    )
    p = Path(path)
    if p.suffix == ".npy":
        np.save(path, np.array(tokens, dtype=np.uint16))
    else:
        with open(path, "wb") as f:
            array.array("H", tokens).tofile(f)
    print(f"  {C.GREEN}✓{C.RESET}  saved {C.CYAN}{len(tokens):,}{C.RESET} "
          f"tokens → {C.DIM}{path}{C.RESET}")


def load_binary(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(path, mmap_mode='r')
    else:
        raw = array.array("H")
        with open(p, "rb") as f:
            raw.fromfile(f, p.stat().st_size // 2)
        arr = np.array(raw, dtype=np.uint16)
    print(f"  {C.GREEN}✓{C.RESET}  loaded {C.CYAN}{len(arr):,}{C.RESET} "
          f"tokens ← {C.DIM}{path}{C.RESET}")
    return arr


def save_npy(tokens: list[int], path: str):
    assert path.endswith(".npy"), "path must end with .npy"
    save_binary(tokens, path)


# ─── split helpers ────────────────────────────────────────────────────────────

def split_npy(src_path: str, train_path: str, val_path: str,
              val_fraction: float = 0.1):
    tokens   = np.load(src_path, mmap_mode='r')
    split_at = int(len(tokens) * (1 - val_fraction))
    np.save(train_path, np.array(tokens[:split_at]))
    np.save(val_path,   np.array(tokens[split_at:]))
    n_train = len(np.load(train_path, mmap_mode='r'))
    n_val   = len(np.load(val_path,   mmap_mode='r'))
    print(f"  {C.DIM}train{C.RESET}  {C.WHITE}{n_train:>12,}{C.RESET} tokens → {C.DIM}{train_path}{C.RESET}")
    print(f"  {C.DIM}val  {C.RESET}  {C.WHITE}{n_val:>12,}{C.RESET} tokens → {C.DIM}{val_path}{C.RESET}")
    return n_train, n_val


def _split_array(arr: np.ndarray, cfg: DataConfig,
                 train_path: str = None,
                 val_path:   str = None):
    n        = len(arr)
    split_at = int(n * cfg.split)

    min_val_tokens = cfg.seq_len + 2
    if n - split_at < min_val_tokens:
        split_at = n - min_val_tokens
        print(f"  {C.YELLOW}⚠  val slice too small — adjusted to "
              f"{split_at/n:.1%} train{C.RESET}")

    train_arr = arr[:split_at]
    val_arr   = arr[split_at:]

    if train_path:
        np.save(train_path, train_arr)
        train_arr = np.load(train_path, mmap_mode='r')
    if val_path:
        np.save(val_path, val_arr)
        val_arr = np.load(val_path, mmap_mode='r')

    train_ds = MemmapTokenDataset.__new__(MemmapTokenDataset)
    train_ds.seq_len   = cfg.seq_len
    train_ds.stride    = cfg.stride or cfg.seq_len
    train_ds.tokens    = train_arr
    train_ds.n_samples = max(0, (len(train_arr) - cfg.seq_len - 1) // train_ds.stride)

    val_ds = MemmapTokenDataset.__new__(MemmapTokenDataset)
    val_ds.seq_len   = cfg.seq_len
    val_ds.stride    = cfg.stride or cfg.seq_len
    val_ds.tokens    = val_arr
    val_ds.n_samples = max(0, (len(val_arr) - cfg.seq_len - 1) // val_ds.stride)

    return train_ds, val_ds


# ─── debug ────────────────────────────────────────────────────────────────────

def _debug_samples(train_dl: DataLoader, cfg: DataConfig,
                   tokenizer: BaseTokenizer = None):
    w = 62
    print(f"\n{C.BOLD}{C.MAGENTA}{'─' * w}{C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}  🔍 Debug samples (train){C.RESET}")
    print(f"{C.DIM}  seq_len={cfg.seq_len}  stride={cfg.stride or cfg.seq_len}  "
          f"batch_size={cfg.batch_size}{C.RESET}")
    print(f"{C.BOLD}{C.MAGENTA}{'─' * w}{C.RESET}\n")

    batch_x, batch_y = next(iter(train_dl))

    n = min(cfg.debug_n, batch_x.shape[0])
    for i in range(n):
        x_ids = batch_x[i].tolist()
        y_ids = batch_y[i].tolist()

        print(f"  {C.BOLD}{C.WHITE}sample {i+1}{C.RESET}")

        x_preview = x_ids[:16]
        y_preview = y_ids[:16]
        tail_x    = f" ... +{len(x_ids)-16}" if len(x_ids) > 16 else ""
        tail_y    = f" ... +{len(y_ids)-16}" if len(y_ids) > 16 else ""
        print(f"  {C.DIM}x ids :{C.RESET} {C.CYAN}{x_preview}{tail_x}{C.RESET}")
        print(f"  {C.DIM}y ids :{C.RESET} {C.CYAN}{y_preview}{tail_y}{C.RESET}")

        if tokenizer is not None:
            try:
                x_text = tokenizer.decode(x_ids)
                y_text = tokenizer.decode(y_ids)
                def _clip(s, n=120):
                    return s[:n] + f"{C.DIM}...{C.RESET}" if len(s) > n else s
                print(f"  {C.DIM}x text:{C.RESET} {C.GREEN}{_clip(repr(x_text))}{C.RESET}")
                print(f"  {C.DIM}y text:{C.RESET} {C.GREEN}{_clip(repr(y_text))}{C.RESET}")
            except Exception as e:
                print(f"  {C.YELLOW}⚠  decode failed — {e}{C.RESET}")

        if x_ids[1:] != y_ids[:-1]:
            print(f"  {C.RED}✗  MISMATCH: y is not x shifted by 1{C.RESET}")
        else:
            print(f"  {C.GREEN}✓  x/y alignment correct{C.RESET}")

        if len(set(x_ids)) == 1:
            print(f"  {C.RED}✗  WARNING: all tokens identical{C.RESET}")

        print()

    print(f"{C.BOLD}{C.MAGENTA}{'─' * w}{C.RESET}\n")


def _debug_stream_info(train_ds: HFStreamDataset, val_ds: HFStreamDataset, val_n: int):
    """Print streaming dataset info (replaces _loaders' sample/batch count for streams)."""
    print(f"  {C.DIM}train{C.RESET}  streaming from row {C.WHITE}{val_n}{C.RESET}+  "
          f"{C.DIM}(infinite){C.RESET}")
    print(f"  {C.DIM}val  {C.RESET}  first {C.WHITE}{val_n}{C.RESET} rows\n")


# ─── loaders ──────────────────────────────────────────────────────────────────

def _loaders(train_ds, val_ds, cfg, shuffle=None,
             tokenizer: BaseTokenizer = None) -> tuple[DataLoader, DataLoader]:
    s        = cfg.shuffle if shuffle is None else shuffle
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=s,
                          num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    print(f"  {C.DIM}train{C.RESET}  {C.WHITE}{len(train_ds):>10,}{C.RESET} samples  "
          f"{C.DIM}│{C.RESET}  {C.YELLOW}{len(train_dl):,}{C.RESET} batches")
    print(f"  {C.DIM}val  {C.RESET}  {C.WHITE}{len(val_ds):>10,}{C.RESET} samples  "
          f"{C.DIM}│{C.RESET}  {C.YELLOW}{len(val_dl):,}{C.RESET} batches\n")
    if cfg.debug:
        _debug_samples(train_dl, cfg, tokenizer)
    return train_dl, val_dl


# ─── tokenize ─────────────────────────────────────────────────────────────────

def _tokenize(texts: list[str], tokenizer: BaseTokenizer,
              bos_id: int = None, eos_id: int = None) -> np.ndarray:
    ids = []
    for i, text in enumerate(texts):
        print(f"\r  {C.DIM}tokenizing{C.RESET}  {_bar(i+1, len(texts))}  "
              f"{C.DIM}{i+1}/{len(texts)}{C.RESET}", end="", flush=True)
        doc = tokenizer.encode(text)
        if bos_id is not None: doc = [bos_id] + doc
        if eos_id is not None: doc = doc + [eos_id]
        ids.extend(doc)
    print()
    return np.array(ids, dtype=np.uint16)


# ─── public API ───────────────────────────────────────────────────────────────

def from_binary(path: str, cfg: DataConfig,
                train_path: str = None,
                val_path:   str = None,
                tokenizer: BaseTokenizer = None) -> tuple[DataLoader, DataLoader]:
    _section("💾 Binary dataset")
    _info("path", path)
    arr              = load_binary(path)
    train_ds, val_ds = _split_array(arr, cfg, train_path, val_path)
    return _loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)


def from_npy_split(train_path: str, val_path: str,
                   cfg: DataConfig,
                   tokenizer: BaseTokenizer = None) -> tuple[DataLoader, DataLoader]:
    _section("💾 Memmap dataset")
    _info("train", train_path)
    _info("val",   val_path)
    train_ds = MemmapTokenDataset(train_path, cfg.seq_len, cfg.stride)
    val_ds   = MemmapTokenDataset(val_path,   cfg.seq_len, cfg.stride)
    return _loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)


def from_files(paths: list[str], tokenizer: BaseTokenizer,
               cfg: DataConfig,
               train_path: str = None,
               val_path:   str = None,
               bos_id: int = None,
               eos_id: int = None) -> tuple[DataLoader, DataLoader]:
    _section("📂 File dataset")
    for p in paths: _info("file", p)

    if cfg.streaming:
        _info("mode", "streaming")
        split_at    = max(1, int(len(paths) * cfg.split))
        train_paths = paths[:split_at]
        val_paths   = paths[split_at:]
        if not val_paths:
            print(f"  {C.YELLOW}⚠  only {len(paths)} file(s) — val overlaps train.{C.RESET}")
            val_paths = train_paths[-1:]
        train_ds = StreamingDataset(train_paths, tokenizer, cfg.seq_len)
        val_ds   = StreamingDataset(val_paths,   tokenizer, cfg.seq_len)
        return _loaders(train_ds, val_ds, cfg, shuffle=False, tokenizer=tokenizer)

    _info("mode", "in-memory")
    texts            = [Path(p).read_text(encoding="utf-8", errors="replace")
                        for p in paths]
    arr              = _tokenize(texts, tokenizer, bos_id, eos_id)
    train_ds, val_ds = _split_array(arr, cfg, train_path, val_path)
    return _loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)


def from_hf(dataset_name: str, tokenizer: BaseTokenizer, cfg: DataConfig,
            split:      str = "train",
            text_col:   str = "text",
            bos_id:     int = None,
            eos_id:     int = None,
            train_path: str = None,
            val_path:   str = None,
            config:     str = None,       # HF dataset config e.g. 'sample-10BT'
            _stream_state: dict = None,   # internal — passed by trainer on resume
            ) -> tuple[DataLoader, DataLoader]:
    """
    Load from a HuggingFace dataset.

    Streaming mode:
      - val = first val_n rows (fixed window)
      - train = everything after val_n rows
      - On resume, pass _stream_state (from checkpoint) to skip already-seen rows

    In-memory mode:
      - Downloads full split, tokenizes, splits train/val
      - Optionally saves .npy for memmap reuse
    """
    _section("🤗 HuggingFace dataset")
    _info("dataset",   dataset_name)
    _info("split",     split)
    _info("streaming", str(cfg.streaming))

    if cfg.streaming:
        val_n      = max(1, int(cfg.batch_size * 20))
        skip_rows  = 0

        # restore resume position if state was passed in
        if _stream_state is not None:
            skip_rows = _stream_state.get("rows_seen", 0)
            prev_chunks = _stream_state.get("chunks_seen", 0)
            prev_tokens = _stream_state.get("tokens_seen", 0)
            print(f"  {C.YELLOW}▶  resuming stream from row {skip_rows:,}  "
                  f"({prev_chunks:,} chunks / {prev_tokens:,} tokens seen){C.RESET}")

        train_ds = HFStreamDataset(
            dataset_name = dataset_name,
            split        = split,
            tokenizer    = tokenizer,
            seq_len      = cfg.seq_len,
            text_col     = text_col,
            bos_id       = bos_id,
            eos_id       = eos_id,
            skip_rows    = val_n + skip_rows,
            take_rows    = None,
            config       = config,
        )
        val_ds = HFStreamDataset(
            dataset_name = dataset_name,
            split        = split,
            tokenizer    = tokenizer,
            seq_len      = cfg.seq_len,
            text_col     = text_col,
            bos_id       = bos_id,
            eos_id       = eos_id,
            skip_rows    = 0,
            take_rows    = val_n,
            config       = config,
        )

        # restore counters onto train_ds (rows_seen is already set via skip_rows above)
        if _stream_state is not None:
            train_ds.chunks_seen = _stream_state.get("chunks_seen", 0)
            train_ds.tokens_seen = _stream_state.get("tokens_seen", 0)

        print(f"  {C.YELLOW}⚠  streaming: val = first {val_n} rows. "
              f"Use in-memory mode for exact splits.{C.RESET}")

        _debug_stream_info(train_ds, val_ds, val_n)

        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

        _ok("streaming dataloaders ready")

        # debug: pull a few samples and show them
        if cfg.debug:
            _debug_samples(train_dl, cfg, tokenizer)

        return train_dl, val_dl

    # ── in-memory ─────────────────────────────────────────────────────────────
    from datasets import load_dataset
    print(f"  {C.YELLOW}⏳ downloading...{C.RESET}", flush=True)
    ds    = load_dataset(dataset_name, split=split, streaming=False)
    texts = [row[text_col] for row in ds]
    _ok(f"downloaded {len(texts):,} documents")
    arr              = _tokenize(texts, tokenizer, bos_id, eos_id)
    train_ds, val_ds = _split_array(arr, cfg, train_path, val_path)
    return _loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)


def from_strings(texts: list[str], tokenizer: BaseTokenizer,
                 cfg: DataConfig,
                 bos_id: int = None,
                 eos_id: int = None) -> tuple[DataLoader, DataLoader]:
    _section("📝 String dataset")
    _info("documents", str(len(texts)))
    arr              = _tokenize(texts, tokenizer, bos_id, eos_id)
    train_ds, val_ds = _split_array(arr, cfg)
    return _loaders(train_ds, val_ds, cfg, tokenizer=tokenizer)