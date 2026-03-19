# chat_template.py
"""
Chat template formatting for SFT training.

No imports from this package — zero circular import risk.

Loss mask rules per assistant turn:
  assistant_header   →  loss=0  (model sees as context)
  response content   →  loss=1
  assistant_closer   →  loss=1  (model must learn to emit this)
  eos_token          →  loss=1, only on the LAST assistant turn
"""


class ChatTemplate:
    """
    Formats a list of {"role": ..., "content": ...} messages into a single
    string and returns character spans where loss should be computed.

    Built-in presets: "chatml", "llama3", "gemma", "alpaca", "raw"

    Special token contract
    ──────────────────────
    Each preset declares which tokens must exist as single vocabulary entries
    via its "special_tokens" list. These must be registered at tokenizer
    train time. Call tok.validate_template(template) at SFT startup.

    EOS placement
    ─────────────
    EOS is appended only after the LAST assistant turn — not between turns.
    This teaches the model to stop after its final response, not after every
    response in a multi-turn conversation.
    """

    PRESETS = {
        "chatml": {
            # role word sandwiched by special tokens — fragmentation irrelevant
            "system_fmt":       "<|im_start|>system<|im_end|>\n{content}<|im_end|>\n",
            "user_fmt":         "<|im_start|>user<|im_end|>\n{content}<|im_end|>\n",
            "assistant_fmt":    "<|im_start|>assistant<|im_end|>\n{content}<|im_end|>\n",
            "assistant_header": "<|im_start|>assistant<|im_end|>\n",   # loss=0
            "assistant_closer": "<|im_end|>\n",                         # loss=1
            "special_tokens":   ["<|im_start|>", "<|im_end|>"],
        },
        "llama3": {
            # role sandwiched between start/end header tokens
            "system_fmt":       "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
            "user_fmt":         "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant_fmt":    "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant_header": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "assistant_closer": "<|eot_id|>",
            "special_tokens":   ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        },
        "gemma": {
            # role sandwiched by start/end of turn tokens
            "system_fmt":       "<start_of_turn>system<end_of_turn>\n{content}<end_of_turn>\n",
            "user_fmt":         "<start_of_turn>user<end_of_turn>\n{content}<end_of_turn>\n",
            "assistant_fmt":    "<start_of_turn>model<end_of_turn>\n{content}<end_of_turn>\n",
            "assistant_header": "<start_of_turn>model<end_of_turn>\n",
            "assistant_closer": "<end_of_turn>\n",
            "special_tokens":   ["<start_of_turn>", "<end_of_turn>"],
        },
        "alpaca": {
            # plain text markers — no special tokens needed
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
        assistant_header : prefix before response content — loss=0
        assistant_closer : suffix after response content — loss=1
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
        self.special_tokens   = (special_tokens if special_tokens is not None
                                 else list(base["special_tokens"]))

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
                              where eos_token is only appended on the LAST
                              assistant turn.

        EOS is embedded as a string inside the span so _align_mask assigns
        it loss=1. Pass tokenizer.decode([eos_id]) as eos_token.
        """
        text           = ""
        response_spans = []

        # find the index of the last assistant turn upfront
        last_assistant_idx = -1
        for idx, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                last_assistant_idx = idx

        for idx, msg in enumerate(messages):
            role    = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                text += self.system_fmt.format(content=content)

            elif role == "user":
                text += self.user_fmt.format(content=content)

            elif role == "assistant":
                header = self.assistant_header
                closer = self.assistant_closer
                # EOS only after the last assistant turn — not between turns
                eos    = eos_token if idx == last_assistant_idx else ""

                span_start = len(text) + len(header)
                span_end   = span_start + len(content) + len(closer) + len(eos)

                text += header + content + closer + eos
                response_spans.append((span_start, span_end))

        return text, response_spans

    def format_single(self, prompt: str, response: str) -> tuple[str, int]:
        """
        Convenience wrapper for single-turn prompt/response pairs.
        Returns (full_text, response_char_start).
        """
        msgs = [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text, spans = self.format_messages(msgs)
        return text, spans[0][0]