"""TINYGSM8K test wrapper — GSM8K math reasoning accuracy.

Two-phase evaluation:
  Phase 1: oga_generate.py — generate model responses for 100 math problems
  Phase 2: evaluate_gsm8k.py — score responses using lm-evaluation-harness
           (exact-match on "#### <number>" answers)

Requires: pip install lm-eval
"""

import os
import re
import sys

from tests.base import BaseTest, TestResult


class TINYGSM8KTest(BaseTest):
    name = "TINYGSM8K"

    def execute(self, model_dir: str, model_params: dict,
                test_params: dict) -> TestResult:
        test_dir = os.path.join(os.path.dirname(__file__), "TINYGSM8K")
        gen_script = os.path.join(test_dir, "oga_generate.py")
        eval_script = os.path.join(test_dir, "evaluate_gsm8k.py")

        context_length = model_params["context_length"]
        output_dir = test_params.get("output_dir", model_dir)

        inputs_file = test_params.get("inputs_file",
                                      os.path.join(test_dir, "tinyGSM8k_inputs_limit-100.json"))
        if not os.path.isabs(inputs_file):
            inputs_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), inputs_file)

        case = test_params.get("case", "psu_prompt_eos_stop")
        max_new_tokens = test_params.get("max_new_tokens", 512)
        eor = test_params.get("eor", "<EOR>")

        responses_file = os.path.join(output_dir, "tinygsm8k_responses.txt")
        eval_output = os.path.join(output_dir, "tinygsm8k_eval.json")

        # --- Phase 1: Generate ---
        print(f"  [{self.name}] Phase 1: Generating responses...")
        gen_cmd = [
            sys.executable, gen_script,
            "-m", model_dir,
            "-i", inputs_file,
            "-o", responses_file,
            "-c", str(context_length),
            "--max-new-tokens", str(max_new_tokens),
            "--case", case,
            "--eor", eor,
            "-v",
        ]
        rc, stdout_gen, stderr_gen = self.run_subprocess(gen_cmd, cwd=test_dir,
                                                         timeout=7200)
        if rc != 0:
            return TestResult(
                success=False, metrics={}, stdout=stdout_gen, stderr=stderr_gen,
                error_msg=f"oga_generate.py exited with code {rc}",
            )

        if not os.path.isfile(responses_file):
            return TestResult(
                success=False, metrics={}, stdout=stdout_gen, stderr=stderr_gen,
                error_msg=f"Responses file not created: {responses_file}",
            )

        # --- Phase 2: Evaluate ---
        print(f"  [{self.name}] Phase 2: Evaluating with lm-eval (gsm8k)...")
        eval_cmd = [
            sys.executable, eval_script,
            "-r", responses_file,
            "--eor", eor,
            "-o", eval_output,
        ]
        rc, stdout_eval, stderr_eval = self.run_subprocess(eval_cmd, cwd=test_dir,
                                                           timeout=600)
        stdout_combined = stdout_gen + "\n--- EVALUATION ---\n" + stdout_eval
        stderr_combined = stderr_gen + "\n" + stderr_eval

        if rc != 0:
            return TestResult(
                success=False, metrics={}, stdout=stdout_combined,
                stderr=stderr_combined,
                error_msg=f"evaluate_gsm8k.py exited with code {rc}",
            )

        metrics = {}
        strict = re.search(r"gsm8k_exact_match_strict=([\d.]+)", stdout_eval)
        flexible = re.search(r"gsm8k_exact_match_flexible=([\d.]+)", stdout_eval)
        if strict:
            metrics["exact_match_strict"] = float(strict.group(1))
        if flexible:
            metrics["exact_match_flexible"] = float(flexible.group(1))

        if not metrics:
            return TestResult(
                success=False, metrics={}, stdout=stdout_combined,
                stderr=stderr_combined,
                error_msg="Could not parse evaluation metrics from output",
            )

        return TestResult(success=True, metrics=metrics,
                          stdout=stdout_combined, stderr=stderr_combined)
