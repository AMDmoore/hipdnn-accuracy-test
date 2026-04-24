#!/usr/bin/env python3
"""Generate per-EP genai_config_*.json variants from a source genai_config.

The repo doesn't ship the model files (model.onnx, prefill-large.onnx, etc.),
so we can't ship the per-EP configs that live next to them either. This script
takes one source genai_config (e.g. the MorphiZenEP-flavored
``genai_config_2k.json`` you already have in your model dir) and emits three
EP-specific siblings:

    genai_config_morphizen_<tag>.json   provider_options=[{"MorphiZenEP": {}}],
                                        session.disable_cpu_ep_fallback="1"
    genai_config_dml_<tag>.json         provider_options=[{"DML": {}}]
    genai_config_cpu_<tag>.json         provider_options=[]            (default
                                        ORT CPU EP)

The graph itself (prefill-large.onnx / decode-large.onnx) is shared across all
three EPs -- it contains only standard ORT contrib ops (GroupQueryAttention,
RotaryEmbedding, SimplifiedLayerNormalization, etc.), no MorphiZen-specific
custom ops -- so the same model files back all three configs.

Usage
-----
    python scripts/build_per_ep_genai_configs.py \
        --source C:/Users/zyq/models/Llama-3.1-8B/genai_config_2k.json \
        --tag    2k

This writes the three files alongside --source and prints the names. Re-run
with a different --source / --tag for additional seq-length buckets.

The script intentionally preserves the rest of the source config
(model.decoder.pipeline, search params, fixed_prompt_length, context_length,
etc.) byte-for-graph-faithful -- we only mutate
``model.decoder.session_options``.
"""

import argparse
import json
import os
import sys


# Per-EP override of model.decoder.session_options. Anything else in the
# source's session_options block is preserved.
EP_PROFILES = {
    "morphizen": {
        "session.disable_cpu_ep_fallback": "1",
        "provider_options": [{"MorphiZenEP": {}}],
    },
    "dml": {
        # DML EP supports CPU fallback for ops it can't run; leave the
        # fallback flag unset so unsupported ops don't crash the session.
        "provider_options": [{"DML": {}}],
    },
    "cpu": {
        # Empty provider_options -> OGA appends no plugin/device EP, ORT
        # falls through to the default CPU EP.
        "provider_options": [],
    },
}


def build_variant(source_cfg: dict, ep_overrides: dict) -> dict:
    """Return a deep-copied source_cfg with session_options replaced for one EP.

    We deliberately *replace* the session_options block (rather than merging)
    so a stale ``session.disable_cpu_ep_fallback`` from the MorphiZen source
    doesn't leak into the CPU/DML variants.
    """
    cfg = json.loads(json.dumps(source_cfg))  # cheap deep copy
    decoder = cfg.setdefault("model", {}).setdefault("decoder", {})
    decoder["session_options"] = dict(ep_overrides)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                        help="Path to source genai_config.json (the MorphiZen "
                             "one is fine; we replace session_options anyway).")
    parser.add_argument("--tag", required=True,
                        help="Suffix for output filenames "
                             "(e.g. '2k' -> genai_config_<ep>_2k.json).")
    parser.add_argument("--out-dir", default=None,
                        help="Destination directory (default: same dir as --source).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files.")
    args = parser.parse_args()

    if not os.path.isfile(args.source):
        sys.exit(f"ERROR: source not found: {args.source}")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.source))
    if not os.path.isdir(out_dir):
        sys.exit(f"ERROR: out-dir does not exist: {out_dir}")

    with open(args.source, "r", encoding="utf-8") as f:
        source_cfg = json.load(f)

    written = []
    for ep, overrides in EP_PROFILES.items():
        out_name = f"genai_config_{ep}_{args.tag}.json"
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path) and not args.force:
            print(f"  skip (exists): {out_name}    (use --force to overwrite)")
            continue
        variant = build_variant(source_cfg, overrides)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(variant, f, indent=4)
            f.write("\n")
        written.append(out_name)
        print(f"  wrote: {out_name}")

    if written:
        print(f"\nWrote {len(written)} file(s) to {out_dir}")
    else:
        print("\nNo files written. Pass --force to overwrite existing variants.")


if __name__ == "__main__":
    main()
