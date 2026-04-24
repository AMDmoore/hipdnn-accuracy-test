"""Configuration utilities for the HipDNN accuracy test framework."""

import json
import os


def load_test_config(config_path: str) -> dict:
    """Load and validate test_config.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "model_dir" not in config:
        raise ValueError("test_config.json must contain 'model_dir'")
    if "genai_configs" not in config:
        raise ValueError("test_config.json must contain 'genai_configs'")
    if "tests" not in config:
        raise ValueError("test_config.json must contain 'tests'")

    model_dir = config["model_dir"]
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    genai_configs = config["genai_configs"]
    for seq_key, cfg_file in genai_configs.items():
        cfg_path = os.path.join(model_dir, cfg_file)
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"genai_config for seq_len={seq_key} not found: {cfg_path}"
            )

    for test_name, test_cfg in config["tests"].items():
        if "seq_lengths" not in test_cfg:
            raise ValueError(f"Test '{test_name}' must have 'seq_lengths' list")
        for sl in test_cfg["seq_lengths"]:
            if str(sl) not in genai_configs:
                raise ValueError(
                    f"Test '{test_name}' references seq_len={sl} "
                    f"but it's not in genai_configs mapping"
                )

    return config


def setup_package_env(package_dir: str, therock_dist: str = None):
    """Configure PATH and env vars from a deployment package directory.

    Expected layout:
        package_dir/
            bin/   — DLLs and executables (onnxruntime.dll, onnxruntime-genai.dll,
                     onnxruntime_morphizen_ep.dll, model_benchmark.exe, etc.)
            lib/   — HIP custom kernels and import libraries

    Sets:
        PATH                — prepends bin/ so DLLs are found at runtime
        HIP_CUSTOM_KERNELS_DIR — points to lib/ for HIP kernel loading
        THEROCK_DIST        — TheRock SDK path (if provided)
    """
    package_dir = os.path.abspath(package_dir)
    bin_dir = os.path.join(package_dir, "bin")
    lib_dir = os.path.join(package_dir, "lib")

    if not os.path.isdir(bin_dir):
        raise FileNotFoundError(f"package bin/ not found: {bin_dir}")
    if not os.path.isdir(lib_dir):
        raise FileNotFoundError(f"package lib/ not found: {lib_dir}")

    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["HIP_CUSTOM_KERNELS_DIR"] = lib_dir

    print(f"Package env : {package_dir}")
    print(f"  bin (PATH): {bin_dir}")
    print(f"  lib (HIP) : {lib_dir}")

    if therock_dist:
        therock_dist = os.path.abspath(therock_dist)
        if not os.path.isdir(therock_dist):
            raise FileNotFoundError(f"therock_dist not found: {therock_dist}")
        os.environ["THEROCK_DIST"] = therock_dist
        therock_bin = os.path.join(therock_dist, "bin")
        if os.path.isdir(therock_bin):
            os.environ["PATH"] = therock_bin + os.pathsep + os.environ.get("PATH", "")
            print(f"  THEROCK   : {therock_dist}")
            print(f"  THEROCK/bin: {therock_bin} (added to PATH)")
        else:
            print(f"  THEROCK   : {therock_dist}")


def extract_seqlen(genai_config_path: str) -> int:
    """Extract the fixed sequence length from a genai_config.json.

    Checks model.decoder.fixed_prompt_length first, then
    model.decoder.sliding_window.window_size.
    Raises ValueError if neither is found.
    """
    with open(genai_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    decoder = config.get("model", {}).get("decoder", {})

    if "fixed_prompt_length" in decoder:
        return int(decoder["fixed_prompt_length"])

    sliding_window = decoder.get("sliding_window", {})
    if "window_size" in sliding_window:
        return int(sliding_window["window_size"])

    raise ValueError(
        f"Cannot extract seqlen from {genai_config_path}: "
        "neither 'fixed_prompt_length' nor 'sliding_window.window_size' found"
    )
