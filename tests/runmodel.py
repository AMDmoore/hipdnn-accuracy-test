"""RUNMODEL test wrapper.

Calls tests/RUNMODEL/run_model.py as a subprocess.
Sets -l (max_length) = seqlen + max_gen_tokens to allow generation.
"""

import json
import os
import sys

from tests.base import BaseTest, TestResult


class RUNMODELTest(BaseTest):
    name = "RUNMODEL"

    def execute(self, model_dir: str, seqlen: int,
                test_params: dict) -> TestResult:
        script = os.path.join(os.path.dirname(__file__), "RUNMODEL", "run_model.py")

        max_gen_tokens = test_params.get("max_gen_tokens", 128)
        max_length = seqlen + max_gen_tokens
        prompt_file = test_params.get("prompt_file", "")

        cmd = [
            sys.executable, script,
            "-m", model_dir,
            "-l", str(max_length),
            "-v",
        ]
        if prompt_file:
            cmd.extend(["-pr", prompt_file])

        output_json = os.path.join(model_dir, "runmodel_output.json")
        cmd.extend(["-o", output_json])

        runmodel_dir = os.path.join(os.path.dirname(__file__), "RUNMODEL")
        rc, stdout, stderr = self.run_subprocess(cmd, cwd=runmodel_dir)

        metrics = {}
        if rc == 0:
            metrics["max_length"] = max_length
            metrics["max_gen_tokens"] = max_gen_tokens

            if os.path.isfile(output_json):
                try:
                    with open(output_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    generations = data.get("generations", [])
                    total_tokens = sum(g.get("tokens", 0) for g in generations)
                    metrics["total_tokens"] = total_tokens
                    metrics["num_prompts"] = len(generations)
                except Exception:
                    pass

            return TestResult(success=True, metrics=metrics,
                              stdout=stdout, stderr=stderr)

        return TestResult(
            success=False, metrics={}, stdout=stdout, stderr=stderr,
            error_msg=f"run_model.py exited with code {rc}",
        )
