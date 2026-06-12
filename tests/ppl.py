"""PPL (Perplexity) test wrapper.

Calls tests/PPL/perplexity.py as a subprocess.

The PPL "window" (``-l``) is just the length of each wikitext2 chunk that
perplexity.py scores (``test_enc[i*seqlen:(i+1)*seqlen]``); it is NOT tied to
any model-side fixed length. The window comes from the orchestrator's
per-iteration ``seq_len`` (injected into ``test_params``); ``seq_lengths`` in
test_config.json is therefore a pure window-size sweep, one dynamic config
serving every entry.
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

        # -l: the wikitext2 chunk window (sequence length scored per chunk).
        seqlen = test_params.get("seq_len")
        if seqlen is None:
            return TestResult(
                success=False, metrics={}, stdout="", stderr="",
                error_msg="PPL requires a window size: set seq_len via "
                          "test_config.json 'seq_lengths'",
            )
        seqlen = int(seqlen)
        nsamples = test_params.get("nsamples", 1.0)

        # -c: OGA search.max_length / KV buffer. Same value as the window here,
        # but distinct semantics — each chunk feeds exactly seqlen tokens, so KV
        # only needs to cover the window. Passing it explicitly avoids OGA
        # falling back to the model's huge architectural context_length (e.g.
        # 131072) and over-allocating.
        context_length = int(test_params.get("context_length", seqlen))
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
