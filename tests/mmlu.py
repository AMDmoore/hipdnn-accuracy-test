"""MMLU test wrapper.

Calls tests/MMLU/test_oga_mmlu.py as a subprocess.
The underlying script uses hardcoded max_length values internally.
"""

import os
import re
import sys

from tests.base import BaseTest, TestResult


class MMLUTest(BaseTest):
    name = "MMLU"

    def execute(self, model_dir: str, model_params: dict,
                test_params: dict) -> TestResult:
        script = os.path.join(os.path.dirname(__file__), "MMLU", "test_oga_mmlu.py")

        seqlen = model_params["seqlen"]
        context_length = model_params["context_length"]

        cmd = [
            sys.executable, script,
            "-m", model_dir,
            "-l", str(seqlen),
            "-c", str(context_length),
        ]

        mmlu_dir = os.path.join(os.path.dirname(__file__), "MMLU")
        rc, stdout, stderr = self.run_subprocess(cmd, cwd=mmlu_dir)

        metrics = {}
        if rc == 0:
            match = re.search(r"AVERAGE ACC:\s*([\d.]+)", stdout)
            if match:
                metrics["average_accuracy"] = float(match.group(1))

            for cat_match in re.finditer(r"(\w+)\s+ACC:\s*([\d.]+)", stdout):
                cat_name = cat_match.group(1).lower()
                metrics[f"accuracy_{cat_name}"] = float(cat_match.group(2))

            if not metrics:
                return TestResult(
                    success=False, metrics={}, stdout=stdout, stderr=stderr,
                    error_msg="Could not parse MMLU accuracy from output",
                )
            return TestResult(success=True, metrics=metrics,
                              stdout=stdout, stderr=stderr)

        return TestResult(
            success=False, metrics={}, stdout=stdout, stderr=stderr,
            error_msg=f"test_oga_mmlu.py exited with code {rc}",
        )
