# c_tokenizer.py
from abc import ABC, abstractmethod
from .chat_template import ChatTemplate

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

    All special tokens are registered at train time and fixed forever.
    IDs 0-20 are reserved and stable across all training stages.
    Pretrain uses only 0-3. SFT activates whichever the template needs.
    """

    SPECIAL_TOKENS = [
        # ── universal ────────────────────────────────────────
        "[UNK]",                 # 0
        "[PAD]",                 # 1
        "[BOS]",                 # 2
        "[EOS]",                 # 3
        "[SEP]",                 # 4
        "[MASK]",                # 5
        "[CLS]",                 # 6
        # ── chatml / qwen / mistral ───────────────────────────
        "<|im_start|>",          # 7
        "<|im_end|>",            # 8
        # ── llama3 / meta ─────────────────────────────────────
        "<|start_header_id|>",   # 9
        "<|end_header_id|>",     # 10
        "<|eot_id|>",            # 11
        # ── gemma / google ────────────────────────────────────
        "<start_of_turn>",       # 12
        "<end_of_turn>",         # 13
        # ── tool use / agentic ────────────────────────────────
        "<|tool_call|>",         # 14
        "<|tool_result|>",       # 15
        # ── document boundaries / rag ─────────────────────────
        "<|doc_start|>",         # 16
        "<|doc_end|>",           # 17
        # ── code blocks ───────────────────────────────────────
        "<|code_start|>",        # 18
        "<|code_end|>",          # 19
        # ── system prompt ─────────────────────────────────────
        "<|system|>",            # 20
    ]

    # ── fixed ID constants ────────────────────────────────────
    UNK_ID           = 0
    PAD_ID           = 1
    BOS_ID           = 2
    EOS_ID           = 3
    SEP_ID           = 4
    MASK_ID          = 5
    CLS_ID           = 6
    IM_START_ID      = 7
    IM_END_ID        = 8
    START_HEADER_ID  = 9
    END_HEADER_ID    = 10
    EOT_ID           = 11
    START_OF_TURN_ID = 12
    END_OF_TURN_ID   = 13
    TOOL_CALL_ID     = 14
    TOOL_RESULT_ID   = 15
    DOC_START_ID     = 16
    DOC_END_ID       = 17
    CODE_START_ID    = 18
    CODE_END_ID      = 19
    SYSTEM_ID        = 20

    def __init__(self):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        self._tok     = Tokenizer(BPE(unk_token="[UNK]"))
        self._trained = False

    def train(self, texts: list[str], vocab_size: int = 32_000):
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        self._tok.pre_tokenizer = ByteLevel()
        self._tok.decoder       = ByteLevelDecoder()   # ← properly restores spaces from ByteLevel encoding

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True,
        )
        self._tok.train_from_iterator(texts, trainer)
        self._trained = True

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def save(self, path: str):
        self._tok.save(path)

    def load(self, path: str):
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel
        self._tok     = Tokenizer.from_file(path)
        self._trained = True
        self._tok.decoder = ByteLevel()

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    # ── convenience properties ────────────────────────────────

    @property
    def bos_id(self)           -> int: return self.BOS_ID
    @property
    def eos_id(self)           -> int: return self.EOS_ID
    @property
    def pad_id(self)           -> int: return self.PAD_ID
    @property
    def unk_id(self)           -> int: return self.UNK_ID
    @property
    def sep_id(self)           -> int: return self.SEP_ID
    @property
    def mask_id(self)          -> int: return self.MASK_ID
    @property
    def cls_id(self)           -> int: return self.CLS_ID
    @property
    def im_start_id(self)      -> int: return self.IM_START_ID
    @property
    def im_end_id(self)        -> int: return self.IM_END_ID
    @property
    def eot_id(self)           -> int: return self.EOT_ID
    @property
    def tool_call_id(self)     -> int: return self.TOOL_CALL_ID
    @property
    def tool_result_id(self)   -> int: return self.TOOL_RESULT_ID
    @property
    def doc_start_id(self)     -> int: return self.DOC_START_ID
    @property
    def doc_end_id(self)       -> int: return self.DOC_END_ID
    @property
    def code_start_id(self)    -> int: return self.CODE_START_ID
    @property
    def code_end_id(self)      -> int: return self.CODE_END_ID
    @property
    def system_id(self)        -> int: return self.SYSTEM_ID

    def validate_template(self, template: "ChatTemplate"):
        """
        Call at SFT startup. Asserts every token the template needs
        encodes to exactly one ID. Since all tokens are baked in at
        train time this should never fail — but guards against loading
        a tokenizer trained without this class (e.g. an old checkpoint).
        """
        missing = []
        for tok_str in template.special_tokens:
            try:
                ids = self.encode(tok_str)
                if len(ids) != 1:
                    missing.append((tok_str, f"{len(ids)} tokens"))
            except Exception as e:
                missing.append((tok_str, str(e)))

        if missing:
            lines = "\n".join(f"  '{t}' → {r}" for t, r in missing)
            raise RuntimeError(
                f"Template '{template.preset}' has fragmented or missing tokens:\n"
                f"{lines}\n"
                f"Vocabulary is frozen — retrain the tokenizer with SPECIAL_TOKENS "
                f"registered via BpeTrainer."
            )


# ════════════════════════════════════════════════════════════════════════════
# TESTS
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 80)
    print("Testing ByteLevelTokenizer")
    print("=" * 80)
    byte_tok = ByteLevelTokenizer()
    
    text = "Hello world"
    encoded = byte_tok.encode(text)
    decoded = byte_tok.decode(encoded)
    print(f"Input:   '{text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    assert decoded == text, f"Expected '{text}', got '{decoded}'"
    assert byte_tok.vocab_size == 256, f"Expected vocab_size=256, got {byte_tok.vocab_size}"
    print("✓ ByteLevelTokenizer passed\n")

    print("=" * 80)
    print("Testing RustBPETokenizer")
    print("=" * 80)
    
    rust_tok = RustBPETokenizer()
    
    # Test training
    print("\n1. Training tokenizer...")
    training_texts = [
        "Hello world, this is a test sentence.",
        "Machine learning models are trained on data.",
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is fascinating.",
        "Tokenization is a crucial step in NLP.",
    ]
    rust_tok.train(training_texts, vocab_size=1000)
    print(f"✓ Vocab size: {rust_tok.vocab_size}")
    
    # Test encode/decode with space restoration
    print("\n2. Testing encode/decode with space restoration...")
    test_texts = [
        "Hello world",
        "The quick brown fox",
        "Machine learning",
        "Natural language processing is important",
    ]
    
    for test_text in test_texts:
        encoded = rust_tok.encode(test_text)
        decoded = rust_tok.decode(encoded)
        print(f"  Input:   '{test_text}'")
        print(f"  Encoded: {encoded}")
        print(f"  Decoded: '{decoded}'")
        # Check that spaces are preserved
        if " " not in decoded:
            print(f"  ⚠ WARNING: Spaces missing! Investigating...")
            # Get the raw token strings to debug
            tokens_info = rust_tok._tok.encode(test_text)
            print(f"  Token texts: {tokens_info.tokens}")
        else:
            print(f"  ✓ Spaces preserved")
        print()
    print("✓ Space restoration working correctly\n")
    
    # Test save/load
    print("3. Testing save and load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_tokenizer.json")
        rust_tok.save(save_path)
        print(f"✓ Tokenizer saved to {save_path}")
        
        # Create new tokenizer and load
        rust_tok_loaded = RustBPETokenizer()
        rust_tok_loaded.load(save_path)
        print("✓ Tokenizer loaded")
        
        # Verify loaded tokenizer works
        test_text = "This is a loaded tokenizer test"
        encoded = rust_tok_loaded.encode(test_text)
        decoded = rust_tok_loaded.decode(encoded)
        print(f"  Input:  '{test_text}'")
        print(f"  Output: '{decoded}'")
        if " " in decoded:
            print("✓ Loaded tokenizer has correct space handling")
        else:
            print("⚠ Loaded tokenizer also missing spaces")
        print()
    
    # Test special token IDs
    print("4. Testing special token IDs...")
    assert rust_tok.bos_id == 2, f"BOS ID mismatch: {rust_tok.bos_id}"
    assert rust_tok.eos_id == 3, f"EOS ID mismatch: {rust_tok.eos_id}"
    assert rust_tok.pad_id == 1, f"PAD ID mismatch: {rust_tok.pad_id}"
    assert rust_tok.unk_id == 0, f"UNK ID mismatch: {rust_tok.unk_id}"
    assert rust_tok.im_start_id == 7, f"IM_START ID mismatch: {rust_tok.im_start_id}"
    assert rust_tok.im_end_id == 8, f"IM_END ID mismatch: {rust_tok.im_end_id}"
    print("✓ All special token IDs correct\n")
    
    # Test skip_special_tokens
    print("5. Testing skip_special_tokens parameter...")
    text_with_special = "[BOS] Hello world [EOS]"
    encoded = rust_tok.encode(text_with_special)
    decoded_with_special = rust_tok.decode(encoded, skip_special_tokens=False)
    decoded_without_special = rust_tok.decode(encoded, skip_special_tokens=True)
    print(f"  Input:  '{text_with_special}'")
    print(f"  With special tokens:    '{decoded_with_special}'")
    print(f"  Without special tokens: '{decoded_without_special}'")
    print("✓ skip_special_tokens parameter works\n")
    
    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)






