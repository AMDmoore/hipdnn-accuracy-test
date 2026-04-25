"""Evaluate TinyGSM8K generated responses by direct gsm8k ground-truth lookup.

Phase 2 of the TINYGSM8K accuracy test: reads pre-generated responses
plus the inputs file used to generate them, looks up each question's
gold answer in gsm8k (train + test), and applies the same regexes
lm-eval's gsm8k task uses (strict-match and flexible-extract).

Earlier versions delegated to lm-eval's task harness, which iterates
its own 1319-question gsm8k test split and pairs the i-th request
with the i-th pre-generated response. That alignment is wrong unless
the inputs file is the gsm8k test split's first N in order, which
``tinyGSM8k_inputs_limit-100.json`` is not, so all metrics reported 0.

Only requires ``datasets`` (already pinned by the test framework).
"""

import argparse
import json
import os
import re
import sys

from datasets import load_dataset

# Same regexes as lm-eval's gsm8k task definition.
STRICT_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEX_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
INVALID = "[invalid]"


def _normalize(s):
    s = s.strip().replace("$", "").replace(",", "").replace("%", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _extract_strict(text):
    m = STRICT_RE.search(text)
    return _normalize(m.group(1)) if m else INVALID


def _extract_flex(text):
    nums = [a or b for a, b in FLEX_RE.findall(text) if (a or b).strip()]
    return _normalize(nums[-1]) if nums else INVALID


def _strip_prompt_wrapper(prompt):
    """Strip 'Question: ... \\nAnswer:' so we can match raw gsm8k question text."""
    s = prompt[len("Question:"):] if prompt.startswith("Question:") else prompt
    return s.split("\nAnswer:")[0].strip()


def _build_gsm8k_lookup():
    """Map question_text -> gold answer across both gsm8k splits."""
    lookup = {}
    for split in ("test", "train"):
        for row in load_dataset("gsm8k", "main", split=split):
            gold = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else INVALID
            lookup[row["question"].strip()] = _normalize(gold)
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyGSM8K responses")
    parser.add_argument("-r", "--responses", type=str, required=True,
                        help="Path to generated responses text file")
    parser.add_argument("-i", "--inputs", type=str, default=None,
                        help="Path to inputs JSON used by oga_generate.py "
                             "(defaults to tinyGSM8k_inputs_limit-100.json "
                             "next to this script)")
    parser.add_argument("--eor", type=str, default="<EOR>",
                        help="End-of-response separator")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    inputs_path = args.inputs or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tinyGSM8k_inputs_limit-100.json",
    )
    with open(inputs_path, "r", encoding="utf-8") as f:
        inputs = json.load(f)

    with open(args.responses, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.endswith(args.eor):
        raw = raw[: -len(args.eor)]
    # Keep empty blocks (an empty response still occupies its slot) so
    # responses[i] stays aligned with inputs[i]. oga_generate.py writes
    # exactly one block per input, so dropping empties would silently
    # misalign every response after the first empty one.
    responses = [b.strip() for b in raw.split(args.eor)]
    n_empty = sum(1 for r in responses if not r)
    print(f"Loaded {len(responses)} pre-generated responses ({n_empty} empty)")

    n = min(len(inputs), len(responses))
    if len(inputs) != len(responses):
        print(f"WARNING: {len(inputs)} inputs vs {len(responses)} responses "
              f"-- scoring first {n}", file=sys.stderr)

    print("Loading gsm8k ground-truth (train + test)...")
    gt_lookup = _build_gsm8k_lookup()

    strict_correct = flex_correct = 0
    for i in range(n):
        q = _strip_prompt_wrapper(inputs[i] if isinstance(inputs[i], str)
                                  else inputs[i].get("question", ""))
        gt = gt_lookup.get(q, INVALID)
        if gt == INVALID:
            continue
        if _extract_strict(responses[i]) == gt:
            strict_correct += 1
        if _extract_flex(responses[i]) == gt:
            flex_correct += 1

    exact_match = strict_correct / n if n else 0.0
    flexible_match = flex_correct / n if n else 0.0

    print(f"\n{'='*50}")
    print(f"GSM8K Results:")
    print(f"  exact_match (strict):   {exact_match:.4f}")
    print(f"  exact_match (flexible): {flexible_match:.4f}")
    print(f"{'='*50}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "task": "gsm8k",
                "exact_match_strict": exact_match,
                "exact_match_flexible": flexible_match,
            }, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")

    print(f"gsm8k_exact_match_strict={exact_match}")
    print(f"gsm8k_exact_match_flexible={flexible_match}")


if __name__ == "__main__":
    main()
