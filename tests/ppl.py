"""PPL (Perplexity) test wrapper.

Calls tests/PPL/perplexity.py as a subprocess.
Uses seqlen as the chunk size (-l) so that max_length matches the
fixed-shape model's prefill length.
"""

import os
import re
import sys

from tests.base import BaseTest, TestResult


class PPLTest(BaseTest):
    name = "PPL"

    def execute(self, model_dir: str, model_params: dict,
                test_params: dict) -> TestResult:
        script = os.path.join(os.path.dirname(__file__), "PPL", "perplexity.py")

        seqlen = model_params["seqlen"]
        context_length = model_params["context_length"]
        nsamples = test_params.get("nsamples", 1.0)

        cmd = [
            sys.executable, script,
            "-m", model_dir,
            "-l", str(seqlen),
            "-c", str(context_length),
            "-n", str(nsamples),
            "-v",
            "-s", "non-raw",
        ]

        rc, stdout, stderr = self.run_subprocess(cmd)

        metrics = {}
        if rc == 0:
            match = re.search(r"Perplexity:\s+([\d.]+)", stdout)
            if match:
                metrics["perplexity"] = float(match.group(1))
            else:
                return TestResult(
                    success=False, metrics={}, stdout=stdout, stderr=stderr,
                    error_msg="Could not parse perplexity from output",
                )
            return TestResult(success=True, metrics=metrics,
                              stdout=stdout, stderr=stderr)

        return TestResult(
            success=False, metrics={}, stdout=stdout, stderr=stderr,
            error_msg=f"perplexity.py exited with code {rc}",
        )
