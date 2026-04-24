"""RUNMODEL test wrapper.

Calls tests/RUNMODEL/run_model.py as a subprocess.
For fixed-shape models, -l (max_length) must equal context_length from
genai_config.json so OGA allocates the correct attention_mask size.
"""

import json
import os
import sys

from tests.base import BaseTest, TestResult


class RUNMODELTest(BaseTest):
    name = "RUNMODEL"

    def execute(self, model_dir: str, model_params: dict,
                test_params: dict) -> TestResult:
        script = os.path.join(os.path.dirname(__file__), "RUNMODEL", "run_model.py")

        context_length = model_params["context_length"]
        max_length = context_length
        prompt_file = test_params.get("prompt_file", "")
        if prompt_file and not os.path.isabs(prompt_file):
            prompt_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), prompt_file)

        max_new_tokens = test_params.get("max_new_tokens", 128)

        cmd = [
            sys.executable, script,
            "-m", model_dir,
            "-l", str(max_length),
            "--max-new-tokens", str(max_new_tokens),
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
            metrics["context_length"] = context_length

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
