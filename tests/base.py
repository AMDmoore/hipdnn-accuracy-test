"""Base class for all accuracy tests."""

import os
import subprocess
import sys
from abc import ABC, abstractmethod


class TestResult:
    """Container for a single test run result."""

    def __init__(self, success: bool, metrics: dict, stdout: str, stderr: str,
                 error_msg: str = ""):
        self.success = success
        self.metrics = metrics
        self.stdout = stdout
        self.stderr = stderr
        self.error_msg = error_msg


class BaseTest(ABC):
    """Base class for accuracy test wrappers.

    The parent run() method handles:
      - Verifying genai_config.json exists in the model dir
      - Creating the per-test output sub-directory
      - Calling the subclass execute()

    All models are dynamic-shape: the per-run sequence-length scale (the OGA
    ``max_length`` / KV-cache cap, and for PPL the wikitext2 chunk window) is
    taken from ``test_params['seq_len']`` (the current sweep value injected by
    run_accuracy.py), NOT from a fixed length in the model's genai_config. So
    run() no longer reads ``decoder.fixed_prompt_length`` / ``sliding_window``.
    """

    name: str = "base"

    def run(self, model_dir: str, test_params: dict) -> TestResult:
        """Verify genai_config exists, create the test output dir, call execute.

        test_params carries an ``output_dir`` key injected by the orchestrator;
        a ``<output_dir>/<test_name>/`` sub-directory is created and stored back
        so ``execute()`` can use it directly. ``seq_len`` (the current sweep
        value) is also injected by the orchestrator.
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

        print(f"  [{self.name}] seq_len={test_params.get('seq_len')}, "
              f"model_dir={model_dir}")

        try:
            # model_params is retained as an extension point (per-model values
            # derived from the graph/config); empty today since every test
            # takes its sequence-length scale from test_params['seq_len'].
            result = self.execute(model_dir, model_params={}, test_params=test_params)
        except Exception as e:
            return TestResult(
                success=False, metrics={}, stdout="", stderr="",
                error_msg=f"execute() raised: {e}",
            )

        return result

    @abstractmethod
    def execute(self, model_dir: str, model_params: dict,
                test_params: dict) -> TestResult:
        """Subclass implements: build command line, run subprocess, parse.

        Args:
            model_dir: path to model directory (contains genai_config.json)
            model_params: reserved extension point for per-model values; empty
                today (every test reads its sequence-length scale from
                test_params['seq_len']).
            test_params: test-specific params from test_config.json, plus
                ``output_dir`` (the ``<top_output_dir>/<test_name>/`` directory
                created by ``run()``) and ``seq_len`` (the current per-iteration
                sequence-length scale injected by the orchestrator).
        """
        ...

    _EP_BOOTSTRAP = os.path.join(os.path.dirname(__file__), "_ep_bootstrap.py")

    def run_subprocess(self, cmd: list, cwd: str = None,
                       timeout: int = 3600) -> tuple:
        """Run a command as subprocess and capture output.

        Transparently injects ``tests/_ep_bootstrap.py`` in front of any
        ``python <target.py> ...`` invocation so plugin execution providers
        (e.g. MorphiZenEP) get registered exactly once, in one place,
        before the target script runs.

        Returns (returncode, stdout, stderr).
        """
        if (len(cmd) >= 2 and isinstance(cmd[1], str)
                and cmd[1].endswith(".py")
                and os.path.isfile(self._EP_BOOTSTRAP)):
            cmd = [cmd[0], self._EP_BOOTSTRAP] + list(cmd[1:])

        print(f"  [{self.name}] Running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return -1, "", str(e)
