"""PPL_VLM (VLM Perplexity) test wrapper.

Calls tests/PPL/perplexity_vlm.py as a subprocess. Differs from the LLM
PPL wrapper in two ways:

1. Overrides ``run()`` to skip ``extract_model_params``. VLM models in our
   zoo (Qwen3.5-9B, Gemma3-4B, etc.) ship pure dynamic-shape genai_configs
   without ``decoder.fixed_prompt_length`` or ``decoder.sliding_window`` —
   the LLM helper raises ``ValueError`` on them. VLM PPL also doesn't need
   a ``seqlen``: chunking is one (image, caption) per sample, not one
   fixed-length text chunk.

2. Forwards VLM-specific test_params (dataset / split / limit / image_size /
   max_length / instruction) to the runner CLI.

The ``test_config.json`` repurposes ``seq_lengths`` as a list of provider
variant names (``allcpu`` / ``allgpu`` / ...). Each entry maps to a
``genai_config_<variant>.json`` file via the existing ``genai_configs``
mapping, so the framework's per-iteration ``switch_genai_config()`` call
swaps the active provider layout in place.
"""

import os
import re
import subprocess
import sys

from tests.base import BaseTest, TestResult


class PPLVLMTest(BaseTest):
    name = "PPL_VLM"

    def run(self, model_dir: str, test_params: dict) -> TestResult:
        """Override: skip extract_model_params; create test output dir; call execute.

        Mirrors BaseTest.run() but does not require fixed_prompt_length /
        sliding_window in genai_config.json — VLM models are dynamic-shape.
        """
        genai_config_path = os.path.join(model_dir, "genai_config.json")
        if not os.path.isfile(genai_config_path):
            return TestResult(
                success=False, metrics={}, stdout="", stderr="",
                error_msg=f"genai_config.json not found at {genai_config_path}",
            )

        top_output_dir = test_params.get("output_dir", "results")
        test_output_dir = os.path.join(top_output_dir, self.name)
        os.makedirs(test_output_dir, exist_ok=True)
        test_params["output_dir"] = test_output_dir

        print(f"  [{self.name}] model_dir={model_dir}")

        try:
            result = self.execute(model_dir, model_params={}, test_params=test_params)
        except Exception as e:
            return TestResult(
                success=False, metrics={}, stdout="", stderr="",
                error_msg=f"execute() raised: {e}",
            )
        return result

    def execute(self, model_dir: str, model_params: dict,
                test_params: dict) -> TestResult:
        script = os.path.join(os.path.dirname(__file__), "PPL", "perplexity_vlm.py")

        dataset = test_params.get("dataset", "lmms-lab/flickr30k")
        split = test_params.get("split", "test")
        limit = int(test_params.get("limit", 50))
        instruction = test_params.get("instruction", "Describe this image briefly.")
        image_size = int(test_params.get("image_size", 896))
        max_length = int(test_params.get("max_length", 384))

        cmd = [
            sys.executable, script,
            "-m", model_dir,
            "-d", dataset,
            "-s", split,
            "-n", str(limit),
            "--instruction", instruction,
            "--image_size", str(image_size),
            "--max_length", str(max_length),
            "-v",
        ]

        # Bypass BaseTest.run_subprocess on purpose: it auto-injects
        # ``_ep_bootstrap.py``, which imports onnxruntime_genai BEFORE our
        # script's top-level ORT-DLL pre-load can run — defeating the
        # workaround for stale pip ORT (1.23 vs the 1.25+ Qwen3.5 needs).
        # perplexity_vlm.py registers MorphiZenEP itself, post-pre-load.
        print(f"  [{self.name}] Running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
                encoding="utf-8",
                errors="replace",
            )
            rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            rc, stdout, stderr = -1, "", "Command timed out after 7200s"
        except Exception as e:
            rc, stdout, stderr = -1, "", str(e)

        metrics = {}
        if rc == 0:
            match = re.search(r"Perplexity:\s+([\d.]+)", stdout)
            if not match:
                return TestResult(
                    success=False, metrics={}, stdout=stdout, stderr=stderr,
                    error_msg="Could not parse Perplexity from output",
                )
            metrics["perplexity"] = float(match.group(1))

            # Also surface the eval/skip ratios so the summary table shows
            # whether samples were silently dropped (e.g. caption too long).
            m2 = re.search(r"Samples evaluated\s*:\s*(\d+)\s*/\s*(\d+)", stdout)
            if m2:
                metrics["samples_evaluated"] = int(m2.group(1))
                metrics["samples_total"] = int(m2.group(2))
            m3 = re.search(r"Total target tokens:\s*(\d+)", stdout)
            if m3:
                metrics["target_tokens"] = int(m3.group(1))
            return TestResult(success=True, metrics=metrics,
                              stdout=stdout, stderr=stderr)

        return TestResult(
            success=False, metrics={}, stdout=stdout, stderr=stderr,
            error_msg=f"perplexity_vlm.py exited with code {rc}",
        )
