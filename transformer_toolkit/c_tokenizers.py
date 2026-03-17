# c_tokenizer.py
from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    @abstractmethod
    def train(self, texts: list[str], vocab_size: int): ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @abstractmethod
    def save(self, path: str): ...

    @abstractmethod
    def load(self, path: str): ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

class ByteLevelTokenizer(BaseTokenizer):
    """
    Zero deps. Every byte is a token (0-255).
    Works on any text/language out of the box.
    """
    def train(self, texts, vocab_size=256): pass   # nothing to train

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path: str): pass
    def load(self, path: str): pass

    @property
    def vocab_size(self) -> int: return 256


class HFTokenizer(BaseTokenizer):
    """
    Thin wrapper around any HuggingFace tokenizer.
    pip install transformers
    """
    def __init__(self, model_name: str = "gpt2"):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(model_name)

    def train(self, texts, vocab_size=None):
        raise NotImplementedError("use HF's train_new_from_iterator for custom training")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def save(self, path: str):
        self._tok.save_pretrained(path)

    def load(self, path: str):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(path)

    @property
    def vocab_size(self) -> int:
        return len(self._tok)
    

class RustBPETokenizer(BaseTokenizer):
    """
    BPE tokenizer backed by HuggingFace's `tokenizers` Rust crate.
    Trains ~100x faster than pure Python BPE.
    pip install tokenizers
    """
    def __init__(self):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        self._tok = Tokenizer(BPE(unk_token="[UNK]"))
        self._trained = False

    def train(self, texts: list[str], vocab_size: int = 8000):
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        self._tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
        )
        self._tok.train_from_iterator(texts, trainer)
        self._trained = True

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def save(self, path: str):
        self._tok.save(path)

    def load(self, path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)
        self._trained = True

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()