"""Base class for all accuracy tests."""

import os
import subprocess
import sys
from abc import ABC, abstractmethod

from config import extract_seqlen


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
      - Reading genai_config.json and extracting seqlen
      - Calling the subclass execute()
      - Capturing stdout/stderr
      - Calling parse_output() to extract metrics
    """

    name: str = "base"

    def run(self, model_dir: str, test_params: dict) -> TestResult:
        """Run the test: extract seqlen, call execute, parse results."""
        genai_config_path = os.path.join(model_dir, "genai_config.json")
        if not os.path.isfile(genai_config_path):
            return TestResult(
                success=False, metrics={}, stdout="", stderr="",
                error_msg=f"genai_config.json not found at {genai_config_path}",
            )

        try:
            seqlen = extract_seqlen(genai_config_path)
        except ValueError as e:
            return TestResult(
                success=False, metrics={}, stdout="", stderr="",
                error_msg=str(e),
            )

        print(f"  [{self.name}] seqlen={seqlen}, model_dir={model_dir}")

        try:
            result = self.execute(model_dir, seqlen, test_params)
        except Exception as e:
            return TestResult(
                success=False, metrics={}, stdout="", stderr="",
                error_msg=f"execute() raised: {e}",
            )

        return result

    @abstractmethod
    def execute(self, model_dir: str, seqlen: int,
                test_params: dict) -> TestResult:
        """Subclass implements: build command line, run subprocess, parse."""
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
