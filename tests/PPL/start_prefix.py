"""Sequence-start prefix for perplexity windows.

Many models expect every sequence to start with a special token (usually BOS)
and score noticeably worse without it. A PPL run slices one long token stream
into many windows, so a BOS that is only present at the head of the stream is
missing from every other window.

The corpus is therefore always encoded with ``add_special_tokens=false`` and
the prefix the tokenizer *would* add is prepended to each window explicitly.
The prefix is discovered by asking the tokenizer itself, so the rule is "give
every window whatever OGA adds at the start of a sequence when special tokens
are enabled". Models whose tokenizer adds nothing get an empty prefix.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

# Any short text that tokenizes to at least two distinct tokens works; the
# probe only compares the two encodings of this text.
PROBE_TEXT = "hello world"

MODES = ("auto", "never")


@dataclass(frozen=True)
class StartPrefix:
    ids: Tuple[int, ...]
    source: str  # "tokenizer" or "disabled"

    def __len__(self) -> int:
        return len(self.ids)


def split_special_tokens(with_special: Sequence[int],
                         without_special: Sequence[int]
                         ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Split ``prefix + without_special + suffix`` into (prefix, suffix).

    Raises if ``without_special`` is not a contiguous run of ``with_special``
    (the tokenizer changed more than the boundaries) or if the position is
    ambiguous."""
    with_special = [int(t) for t in with_special]
    without_special = [int(t) for t in without_special]
    n = len(without_special)
    extra = len(with_special) - n
    offsets = [] if extra < 0 else [
        p for p in range(extra + 1) if with_special[p:p + n] == without_special
    ]
    if len(offsets) != 1:
        raise ValueError(
            "cannot derive the start prefix: add_special_tokens=false encoding "
            f"{without_special} occurs {len(offsets)} times in the "
            f"add_special_tokens=true encoding {with_special}")
    p = offsets[0]
    return tuple(with_special[:p]), tuple(with_special[p + n:])


def detect_start_prefix(tokenizer) -> Tuple[int, ...]:
    """Ids the OGA tokenizer prepends when ``add_special_tokens`` is enabled.

    Trailing special tokens (e.g. EOS) are discarded: windows end mid-text, so
    only the sequence start applies. Leaves the tokenizer with
    ``add_special_tokens`` disabled."""
    try:
        tokenizer.update_options(add_special_tokens="true")
        with_special = tokenizer.encode(PROBE_TEXT)
    finally:
        tokenizer.update_options(add_special_tokens="false")
    without_special = tokenizer.encode(PROBE_TEXT)
    prefix, _ = split_special_tokens(with_special, without_special)
    return prefix


def resolve_start_prefix(tokenizer, mode: str = "auto") -> StartPrefix:
    if mode not in MODES:
        raise ValueError(f"unknown bos mode {mode!r}, expected one of {MODES}")
    if mode == "never":
        return StartPrefix(ids=(), source="disabled")
    return StartPrefix(ids=detect_start_prefix(tokenizer), source="tokenizer")
