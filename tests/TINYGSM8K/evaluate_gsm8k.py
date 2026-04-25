"""Evaluate TinyGSM8K generated responses using lm-evaluation-harness offline mode.

Phase 2 of the TINYGSM8K accuracy test: reads pre-generated responses,
wraps them in a minimal lm_eval LM interface, and runs the gsm8k task
to compute exact-match accuracy.

Only requires ``lm-eval`` — no quark, optimum, or other heavy dependencies.
"""

import argparse
import json
import os
import sys

from lm_eval import evaluator
from lm_eval.api.model import LM


class OfflineGenWrapper(LM):
    """Wraps pre-generated text responses as an lm_eval model.

    Only ``generate_until`` is implemented since gsm8k evaluation only
    needs greedy generation output (no log-likelihood scoring).
    """

    def __init__(self, outputs_path: str, eor: str = "<EOR>"):
        super().__init__()
        with open(outputs_path, "r", encoding="utf-8") as f:
            raw = f.read().strip().rstrip(eor).split(eor)
        self._outputs = [r.strip() for r in raw if r.strip()]
        print(f"Loaded {len(self._outputs)} pre-generated responses")

    def generate_until(self, requests, disable_tqdm=False):
        if len(self._outputs) != len(requests):
            print(f"WARNING: {len(self._outputs)} outputs vs {len(requests)} "
                  f"requests — truncating to min")
        n = min(len(self._outputs), len(requests))

        results = []
        for i in range(n):
            resp = self._outputs[i]
            until_tokens = requests[i].args[1].get("until", [])
            for tok in until_tokens:
                if tok in resp:
                    resp = resp.split(tok)[0]
            results.append(resp)

        if len(results) < len(requests):
            results.extend([""] * (len(requests) - len(results)))
        return results

    def loglikelihood(self, requests, disable_tqdm=False):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyGSM8K responses")
    parser.add_argument("-r", "--responses", type=str, required=True,
                        help="Path to generated responses text file")
    parser.add_argument("--eor", type=str, default="<EOR>",
                        help="End-of-response separator")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of evaluation samples")
    parser.add_argument("--num-fewshot", type=int, default=None,
                        help="Number of few-shot examples (default: task default)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    model = OfflineGenWrapper(args.responses, eor=args.eor)

    results = evaluator.simple_evaluate(
        model=model,
        tasks=["gsm8k"],
        limit=args.limit,
        num_fewshot=args.num_fewshot,
    )

    task_results = results.get("results", {}).get("gsm8k", {})
    exact_match = task_results.get("exact_match,strict-match", 0.0)
    flexible_match = task_results.get("exact_match,flexible-extract", 0.0)

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
                "raw_results": task_results,
            }, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")

    print(f"gsm8k_exact_match_strict={exact_match}")
    print(f"gsm8k_exact_match_flexible={flexible_match}")


if __name__ == "__main__":
    main()
