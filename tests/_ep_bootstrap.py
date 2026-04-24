"""Subprocess wrapper: register plugin EPs, then exec the target as __main__.

Centralizes ONNX Runtime GenAI execution-provider registration in one place
so individual test scripts don't have to. ``BaseTest.run_subprocess`` injects
this wrapper in front of every subprocess test invocation:

    [python, target.py, *args]
        becomes
    [python, _ep_bootstrap.py, target.py, *args]

The target sees the same ``sys.argv`` and ``sys.path[0]`` it would have if
launched directly, so existing relative-import patterns (e.g.
``from tokenizer_factory import ...``) keep working unchanged.

Plugin EP DLLs are discovered on ``PATH`` (which the orchestrator populates
via ``config.setup_package_env``). Add new EPs by appending to ``EP_DLLS``.
"""

import os
import runpy
import sys

EP_DLLS = {
    "MorphiZenEP": "onnxruntime_morphizen_ep.dll",
}


def _register_plugin_eps():
    try:
        import onnxruntime_genai as og
    except ImportError:
        return
    if not hasattr(og, "register_execution_provider_library"):
        return
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for ep_name, dll_name in EP_DLLS.items():
        for d in path_dirs:
            if not d:
                continue
            candidate = os.path.join(d, dll_name)
            if os.path.isfile(candidate):
                try:
                    og.register_execution_provider_library(ep_name, candidate)
                    print(f"[_ep_bootstrap] Registered plugin EP: "
                          f"{ep_name} -> {candidate}")
                except Exception as e:
                    print(f"[_ep_bootstrap] WARN: failed to register "
                          f"{ep_name}: {e}", file=sys.stderr)
                break


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python _ep_bootstrap.py <target.py> [args...]")

    target = os.path.abspath(sys.argv[1])
    if not os.path.isfile(target):
        sys.exit(f"_ep_bootstrap: target not found: {target}")

    sys.argv = sys.argv[1:]
    sys.path[0] = os.path.dirname(target)

    _register_plugin_eps()

    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
