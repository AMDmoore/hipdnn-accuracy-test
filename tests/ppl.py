"""PPL (Perplexity) test wrapper.

Calls tests/PPL/perplexity.py as a subprocess.

The PPL "window" (``-l``) is just the length of each wikitext2 chunk that
perplexity.py scores (``test_enc[i*seqlen:(i+1)*seqlen]``); it is NOT tied to
any model-side fixed length. Dynamic-shape models ship a genai_config WITHOUT
``decoder.fixed_prompt_length`` / ``decoder.sliding_window`` — so this wrapper
takes the window straight from the orchestrator's per-iteration ``seq_len``
(injected into ``test_params``) and skips ``extract_model_params`` entirely
(mirrors ``PPLVLMTest``). ``seq_lengths`` in test_config.json is therefore a
pure window-size sweep; one dynamic config serves every entry.
"""

import os
import re
import sys

from tests.base import BaseTest, TestResult


class PPLTest(BaseTest):
    name = "PPL"

    def run(self, model_dir: str, test_params: dict) -> TestResult:
        """Override BaseTest.run: skip extract_model_params (dynamic-shape).

        The window size comes from ``test_params['seq_len']`` (the current
        sweep value injected by run_accuracy.py), not from a fixed length in
        the model's genai_config.
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

        print(f"  [{self.name}] window(seq_len)={test_params.get('seq_len')}, "
              f"model_dir={model_dir}")

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
        script = os.path.join(os.path.dirname(__file__), "PPL", "perplexity.py")

        # PPL window: the wikitext2 chunk length, and the ONLY knob PPL needs.
        # Driven by the per-iteration sweep value (test_config 'seq_lengths').
        seqlen = test_params.get("seq_len")
        if seqlen is None:
            return TestResult(
                success=False, metrics={}, stdout="", stderr="",
                error_msg="PPL requires a window size: set seq_len via "
                          "test_config.json 'seq_lengths'",
            )
        seqlen = int(seqlen)
        nsamples = test_params.get("nsamples", 1.0)

        # -c (OGA search.max_length / KV buffer) is set equal to the window:
        # each chunk feeds exactly seqlen tokens, so KV only needs to cover the
        # window. Passing it explicitly avoids OGA falling back to the model's
        # huge architectural context_length (e.g. 131072) and over-allocating.
        cmd = [
            sys.executable, script,
            "-m", model_dir,
            "-l", str(seqlen),
            "-c", str(seqlen),
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
