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
