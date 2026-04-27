#!/usr/bin/env python3
"""HipDNN Accuracy Test Framework — Main Orchestrator.

Usage:
    python run_accuracy.py --config test_config.json --tests PPL MMLU
    python run_accuracy.py --tests PPL --seq-len 1024 2048
    python run_accuracy.py --model-dir D:/path/to/model --tests RUNMODEL
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

from config import load_test_config, setup_package_env
from results.reporter import ResultCollector
from tests.ppl import PPLTest
from tests.mmlu import MMLUTest
from tests.runmodel import RUNMODELTest
from tests.tinygsm8k import TINYGSM8KTest

TEST_REGISTRY = {
    "PPL": PPLTest,
    "MMLU": MMLUTest,
    "RUNMODEL": RUNMODELTest,
    "TINYGSM8K": TINYGSM8KTest,
}


def switch_genai_config(model_dir: str, config_filename: str):
    """Copy the seq-specific genai_config to genai_config.json."""
    src = os.path.join(model_dir, config_filename)
    dst = os.path.join(model_dir, "genai_config.json")
    print(f"  Switching genai_config: {config_filename} -> genai_config.json")
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="HipDNN Accuracy Test Framework",
    )
    parser.add_argument(
        "--config", type=str, default="test_config.json",
        help="Path to test_config.json (default: test_config.json)",
    )
    parser.add_argument(
        "--package-dir", type=str, default=None,
        help="Path to deployment package (bin/ + lib/). "
             "Overrides package_dir from config.",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Override model_dir from config",
    )
    parser.add_argument(
        "--tests", nargs="+", default=None,
        help="Tests to run (e.g. PPL MMLU RUNMODEL). "
             "If not specified, runs all tests listed in config.",
    )
    parser.add_argument(
        "--seq-len", nargs="+", type=int, default=None,
        help="Filter specific seq_lengths to run (default: all in config)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Exact output directory for this run's results. Overrides "
             "the per-run subfolder derived from config['output_dir'].",
    )
    args = parser.parse_args()

    config = load_test_config(args.config)

    package_dir = args.package_dir or config.get("package_dir")
    therock_dist = config.get("therock_dist")
    if package_dir:
        setup_package_env(package_dir, therock_dist=therock_dist)
    print()

    model_dir = args.model_dir or config["model_dir"]
    genai_configs = config["genai_configs"]
    tests_config = config["tests"]

    requested_tests = args.tests or list(tests_config.keys())
    for t in requested_tests:
        if t.upper() not in TEST_REGISTRY:
            print(f"ERROR: Unknown test '{t}'. Available: {list(TEST_REGISTRY.keys())}")
            sys.exit(1)

    model_name = os.path.basename(os.path.normpath(model_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_output_dir = config.get("output_dir", "results")
        output_dir = os.path.join(base_output_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    collector = ResultCollector(output_dir, model_name)

    print(f"Model dir   : {model_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Tests       : {requested_tests}")
    print()

    for test_name in requested_tests:
        test_name_upper = test_name.upper()
        test_cfg = tests_config.get(test_name_upper, {})
        test_params = test_cfg.get("params", {})
        seq_lengths = test_cfg.get("seq_lengths", [])

        if args.seq_len:
            seq_lengths = [s for s in seq_lengths if s in args.seq_len]

        if not seq_lengths:
            print(f"[{test_name_upper}] No matching seq_lengths to run, skipping.")
            continue

        test_cls = TEST_REGISTRY[test_name_upper]
        test_instance = test_cls()

        for sl in seq_lengths:
            config_file = genai_configs[str(sl)]
            print(f"[{test_name_upper}] seq_len={sl}, config={config_file}")

            switch_genai_config(model_dir, config_file)

            result = test_instance.run(model_dir, test_params)

            collector.record(
                test_name=test_name_upper,
                seq_len=sl,
                config_file=config_file,
                metrics=result.metrics,
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.success,
                error_msg=result.error_msg,
            )

            status = "PASS" if result.success else "FAIL"
            print(f"  => {status}")
            if result.success and result.metrics:
                for k, v in result.metrics.items():
                    print(f"     {k}: {v}")
            if not result.success:
                print(f"     Error: {result.error_msg}")
            print()

    collector.write_summary()


if __name__ == "__main__":
    main()
