"""Local tokenizer factory for run_model.py.

Provides get_tokenizer() that returns an OGA tokenizer wrapping
AutoTokenizer for encode and using og.Tokenizer for decode.
"""

import onnxruntime_genai as og
from transformers import AutoTokenizer


class OGATokenizerWrapper:
    """Wraps OGA Tokenizer and HF AutoTokenizer together."""

    def __init__(self, model_path, og_model):
        self._og_tokenizer = og.Tokenizer(og_model)
        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            model_path, token=False, use_fast=True, trust_remote_code=True
        )

    def encode(self, text):
        return self._og_tokenizer.encode(text)

    def decode(self, tokens):
        return self._og_tokenizer.decode(tokens)


def get_tokenizer(model_path, model_type, og_model):
    """Return a tokenizer suitable for the given model type."""
    return OGATokenizerWrapper(model_path, og_model)
