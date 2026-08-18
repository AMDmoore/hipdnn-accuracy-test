# HipDNN Accuracy Test Framework

An **LLM / VLM accuracy regression test framework** for the AMD GPU execution provider (hip-ep).

It drives a set of standard accuracy benchmarks (PPL, MMLU, GSM8K, etc.) through [ONNX Runtime GenAI (OGA)](https://github.com/microsoft/onnxruntime-genai), then collects the results into CSV / JSON reports. Use it to verify whether a given deployment package meets the accuracy bar on a given model, and to catch regressions.

---

## Table of Contents

- [What it does](#what-it-does)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration file `test_config.json`](#configuration-file-test_configjson)
- [Test parameters](#test-parameters)
- [Running tests](#running-tests)
- [Results](#results)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## What it does

The framework ships with the following tests. Each one is an accuracy metric computed on the model output produced by OGA:

| Test | Model type | Metric | Description |
|---|---|---|---|
| `PPL` | LLM (text) | perplexity | Perplexity over wikitext2 using fixed-size chunks; lower is better |
| `MMLU` | LLM (text) | average accuracy | Multi-subject multiple-choice benchmark; higher is better |
| `TINYGSM8K` | LLM (text) | exact match | GSM8K math reasoning (100-question subset); requires `lm-eval` |
| `RUNMODEL` | LLM (text) | generated tokens / output | Text generation from a given prompt; produces inspectable output |
| `PPL_VLM` | VLM (multimodal) | perplexity | Vision-language conditional perplexity over (image, caption) pairs |

`PPL` / `MMLU` / `TINYGSM8K` / `RUNMODEL` are **text-only LLM tests**; VLM models only support `PPL_VLM`.

Everything is orchestrated by `run_accuracy.py`. You only edit a single configuration file to select the model, the package, and which tests to run.

---

## Prerequisites

| Dependency | Notes |
|---|---|
| OS | Windows (scripts target PowerShell 5.1) |
| Python | **3.10** (verified; must match the OGA version) |
| Deployment package | Directory with `bin/` (DLLs + exe) and `lib/` (HIP custom kernels) |
| OGA runtime | Directory with `onnxruntime_genai.cp310-win_amd64.pyd` and `onnxruntime-genai.dll` |
| TheRock SDK | ROCm SDK directory (with `bin/`, `lib/`) |
| ONNX model directory | OGA-format model with `model.onnx` and `genai_config*.json` |

Have the paths to these directories ready in advance: the OGA runtime, the deployment package, the TheRock SDK, and the model directory.

---

## Setup

Create an isolated Python 3.10 environment and install the pinned dependencies:

```powershell
# 1. Create a venv with Python 3.10
py -3.10 -m venv C:\work\venv310
C:\work\venv310\Scripts\Activate.ps1

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install `
    --index-url https://pypi.org/simple `
    --extra-index-url https://download.pytorch.org/whl/cpu `
    transformers==5.6.2 tokenizers==0.22.2 huggingface_hub==1.11.0 `
    safetensors==0.7.0 regex==2026.4.4 numpy==2.4.4 pandas==3.0.2 `
    colorama==0.4.6 tqdm==4.67.3 thefuzz==0.22.1 onnx==1.21.0 `
    torch==2.11.0+cpu datasets

# 3. (TINYGSM8K only) install lm-evaluation-harness
python -m pip install lm-eval
```

Then make the local OGA build importable in the venv so that `import onnxruntime_genai` resolves to it. Any one of the following works:

1. **Install the OGA wheel** — if you have a built wheel, run `pip install onnxruntime_genai-*.whl`.
2. **Point at the `.pyd` + DLL** — add the OGA runtime directory to `PYTHONPATH` and register it with `os.add_dll_directory` so the adjacent `onnxruntime-genai.dll` loads.
3. **Copy into `site-packages`** — copy `onnxruntime_genai.cp310-win_amd64.pyd` and `onnxruntime-genai.dll` into the venv's `site-packages`.

---

## Configuration file `test_config.json`

This is the core configuration. **LLM and VLM use different config styles** — the top-level fields are the same, but the meaning of `genai_configs` / `seq_lengths` differs, and the tests are mutually exclusive (LLM cannot run `PPL_VLM`, and VLM can only run `PPL_VLM`).

Top-level fields shared by all configs:

| Field | Meaning |
|---|---|
| `model_dir` | OGA model directory |
| `package_dir` | deployment package directory (overridable via `--package-dir`) |
| `therock_dist` | TheRock SDK directory |
| `output_dir` | base output directory; actual results go to `<output_dir>/<model_name>_<timestamp>/` |
| `genai_configs` | **key → genai_config file** mapping. Before each entry, the framework copies the mapped file to `genai_config.json` (the key's meaning differs for LLM/VLM, see below) |
| `tests` | set of tests to run; each test has a `seq_lengths` list and a `params` object whose meaning differs for LLM/VLM (see below) |

`package_dir` and `therock_dist` must point at the deployment package and the TheRock SDK — this is where the package and TheRock DLL paths get configured. At runtime `setup_package_env()` in `run_accuracy.py` uses them to wire the DLL search path automatically.

### LLM config (PPL / MMLU / RUNMODEL / TINYGSM8K)

```json
{
    "model_dir": "D:/models/Llama-3.1-8B-awq-int4-onnx",
    "package_dir": "D:/pkgs/package_gfx1151_therock7.11",
    "therock_dist": "C:/workspace/therock",
    "output_dir": "results",

    "genai_configs": {
        "2048": "genai_config.json",
        "4096": "genai_config.json"
    },

    "tests": {
        "PPL": {
            "seq_lengths": [2048],
            "params": { "nsamples": 0.1, "stride": 256 }
        }
    }
}
```

Per-test fields for LLM:

| Field | Meaning |
|---|---|
| `tests.<NAME>.seq_lengths` | **sequence-length sweep list** (all models are treated as dynamic-shape). For `PPL` it is the wikitext2 chunk window; for `MMLU` the input-length cap; for `RUNMODEL` / `TINYGSM8K` the OGA `max_length` (KV-cache cap, must be ≥ `prompt_len + max_new_tokens`). Each value **must exist as a key in `genai_configs`** |
| `tests.<NAME>.params` | per-test parameters (see [Test parameters](#test-parameters)) |

If a dynamic model has only one config, map every length to the same file, e.g. `{"2048": "genai_config.json", "4096": "genai_config.json"}`, to keep the mapping self-documenting.

### VLM config (`PPL_VLM` only)

```json
{
    "model_dir": "D:/models/Qwen3.5-9B-vl-onnx",
    "package_dir": "D:/pkgs/package_gfx1151_therock7.11",
    "therock_dist": "C:/workspace/therock",

    "genai_configs": {
        "allgpu": "genai_config_allgpu.json",
        "allcpu": "genai_config_allcpu.json"
    },

    "tests": {
        "PPL_VLM": {
            "seq_lengths": ["allgpu"],
            "params": {
                "dataset": "lmms-lab/flickr30k",
                "split": "test",
                "limit": 50,
                "image_size": 896,
                "max_length": 1024,
                "instruction": "Describe this image briefly."
            }
        }
    }
}
```

Per-test fields for VLM:

| Field | Meaning |
|---|---|
| `PPL_VLM.seq_lengths` | **not a sequence length for VLM.** Repurposed as a list of "provider variant" names (e.g. `allgpu` / `allcpu`), each mapping to a `genai_config_<variant>.json`. For each name the framework swaps the mapped config into `genai_config.json`, switching the EP layout for that run (all-GPU / all-CPU / etc.). Each value **must exist as a key in `genai_configs`** |
| `PPL_VLM.params` | per-test parameters (see [Test parameters](#test-parameters)) |

Notes:

- **VLM does not use fixed-size chunking; the prefill length varies per sample.** Each sample's sequence = prompt (chat template + image tokens + instruction) + caption text. The image-token count is fixed by `image_size` (the image is resized to a square, so it produces the same number of tokens every sample), and only the caption length varies.
- `params.max_length` is the OGA KV-buffer cap. It matters because the genai_config default (e.g. `262144` with `past_present_share_buffer: true`) would pre-allocate an enormous KV cache; setting `max_length` shrinks that to a workable size. It is also the per-sample inclusion threshold: any sample whose full length exceeds `max_length` is **skipped**. So `max_length` must be large enough to fit `prompt (≈ image tokens) + caption`; raise it (not `seq_lengths`) to admit longer samples.

### Bundled config references

- `test_config.json` — LLM tests (PPL / MMLU / RUNMODEL / TINYGSM8K)
- `test_config_vlm.json` — VLM perplexity test (`PPL_VLM`)

---

## Test parameters

`params` supported by each test:

### PPL (perplexity)

| Parameter | Default | Meaning |
|---|---|---|
| `nsamples` | `1.0` | sampling ratio/count (e.g. `0.1` = 10%); smaller is faster |
| `stride` | none | sliding-window stride; only applies to the pruned (last-position-only logits) path |

### MMLU (multi-subject multiple choice)

No extra required parameters; the sequence length comes from `seq_lengths`.

### RUNMODEL (text generation)

| Parameter | Default | Meaning |
|---|---|---|
| `prompt_file` | none | prompt file path (relative to project root, e.g. `tests/RUNMODEL/prompts/prompt_2k.txt`) |
| `max_new_tokens` | `128` | max new tokens to generate |

### TINYGSM8K (math reasoning)

| Parameter | Default | Meaning |
|---|---|---|
| `case` | `psu_prompt_eos_stop` | prompt / stop-condition combination |
| `max_new_tokens` | `512` | max new tokens to generate |
| `inputs_file` | built-in 100 questions | input questions JSON |
| `eor` | `<EOR>` | end-of-response marker |

### PPL_VLM (vision-language perplexity)

| Parameter | Default | Meaning |
|---|---|---|
| `dataset` | `lmms-lab/flickr30k` | dataset |
| `split` | `test` | split |
| `limit` | `50` | number of samples |
| `image_size` | `896` | square side length the image is resized to (also fixes the image-token count) |
| `max_length` | `1024` | OGA KV-buffer cap; also the per-sample inclusion threshold (samples longer than this are skipped). |
| `instruction` | `Describe this image briefly.` | prompt instruction |

---

## Running tests

After activating the venv, run from the project root:

```powershell
# LLM: run all tests listed in the config (default test_config.json)
python run_accuracy.py --config test_config.json

# LLM: run only specific tests
python run_accuracy.py --tests PPL

# LLM: run only specific sequence lengths
python run_accuracy.py --tests PPL --seq-len 2048

# VLM: use the VLM config; only PPL_VLM is supported
python run_accuracy.py --config test_config_vlm.json --tests PPL_VLM

# Override model / package / output directory from the command line
python run_accuracy.py --model-dir D:/models/other --package-dir D:/pkgs/xxx --output-dir results/my_run
```

Command-line arguments:

| Argument | Meaning |
|---|---|
| `--config` | config file path (default `test_config.json`) |
| `--tests` | tests to run (multiple allowed, space-separated). LLM: `PPL` / `MMLU` / `RUNMODEL` / `TINYGSM8K`; VLM: `PPL_VLM` only. If omitted, runs everything in the config |
| `--seq-len` | run only the given sequence lengths (filtered from the config's `seq_lengths`) |
| `--model-dir` | override `model_dir` from the config |
| `--package-dir` | override `package_dir` from the config |
| `--output-dir` | exact output directory for this run |

---

## Results

Each run produces the following under `<output_dir>/<model_name>_<timestamp>/`:

```
<model_name>_<timestamp>/
├── results_summary.csv     # flat metric summary (one metric per row)
├── results_detail.json     # full detail (success/failure, error message, log file name)
└── logs/
    └── <test>_seq<length>_<timestamp>.log   # full stdout / stderr per sub-test
```

- `results_summary.csv` columns: `model_name, seq_len, config_file, test, metric, value, timestamp`. Convenient for importing into a spreadsheet for comparison.
- `results_detail.json` keeps the success/failure and error message of each sub-test; check this first when diagnosing a failure.
- The full output of a single test (for locating an error) is in the corresponding `.log` file under `logs/`.

The console also prints each test's `PASS` / `FAIL` and key metrics in real time.

---

## Project structure

```
hipdnn-accuracy-test/
├── run_accuracy.py          # main orchestrator (select model/package/tests, loop, aggregate)
├── config.py                # load/validate test_config.json; set up package env vars
├── test_config.json         # default config (LLM tests)
├── test_config_vlm.json     # VLM perplexity config example
├── tests/
│   ├── base.py              # base class for all tests (subprocess exec + result wrapping)
│   ├── ppl.py / ppl_vlm.py  # PPL / VLM PPL wrappers
│   ├── mmlu.py              # MMLU wrapper
│   ├── runmodel.py          # text-generation wrapper
│   ├── tinygsm8k.py         # GSM8K two-phase (generate + score) wrapper
│   ├── PPL/ MMLU/ RUNMODEL/ TINYGSM8K/   # underlying scripts and data per test
└── results/
    └── reporter.py          # result collection; writes CSV / JSON / per-test logs
```

Call chain of a single run: `run_accuracy.py` → load config → `setup_package_env()` wires DLLs → for each `(test, seq_len)` combination, swap `genai_config` → call the matching `tests/*.py` wrapper → the wrapper runs the underlying script as a subprocess and parses metrics → `reporter.py` aggregates and writes to disk.

---

## Troubleshooting

**Q: `import onnxruntime_genai` fails with a version/ABI error.**
The OGA `.pyd` is `cp310` ABI, so the venv must use Python 3.10. Recreate the venv with a 3.10 interpreter.

**Q: `import onnxruntime_genai` fails / DLL not found.**
Make sure the OGA runtime directory contains both `onnxruntime_genai.cp310-win_amd64.pyd` and `onnxruntime-genai.dll` and is importable (on `PYTHONPATH` / `os.add_dll_directory`), and that `package_dir` and `therock_dist` in `test_config.json` are correct so their `bin\` directories get onto the DLL search path.

**Q: TINYGSM8K phase 2 fails with gsm8k / lm_eval not found.**
Run `pip install lm-eval` first (it is not part of the default dependencies).

**Q: A `seq_len` reports "not in genai_configs mapping".**
Every value in `tests.<NAME>.seq_lengths` must have a matching key in the top-level `genai_configs`. A dynamic model can point multiple keys at the same config file.

**Q: RUNMODEL / TINYGSM8K generation is truncated.**
Increase `seq_lengths` (the OGA `max_length`) so it is ≥ `prompt_len + max_new_tokens`.
