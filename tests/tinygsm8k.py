"""TINYGSM8K test wrapper — currently shelved.

This test requires a separate Conda environment ("step2") and additional
dependencies (nltk, lm_eval, optimum, tinyBenchmarks, numpy==1.26.4).
The two-phase flow is:
  Phase 1: oga_responses_generation.py  (runs in the main env)
  Phase 2: llm_eval.py --mode offline   (runs in the step2 conda env)
The interface is reserved for future implementation.
"""

from tests.base import BaseTest, TestResult


class TINYGSM8KTest(BaseTest):
    name = "TINYGSM8K"

    def execute(self, model_dir: str, seqlen: int,
                test_params: dict) -> TestResult:
        return TestResult(
            success=False,
            metrics={},
            stdout="",
            stderr="",
            error_msg="TINYGSM8K is currently shelved — requires separate Conda env",
        )
