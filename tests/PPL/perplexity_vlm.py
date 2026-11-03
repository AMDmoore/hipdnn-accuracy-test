"""VLM (Vision-Language Model) Perplexity runner.

Computes conditional perplexity PPL(caption | image) over a small image+caption
dataset (default: HuggingFace ``lmms-lab/flickr30k`` test split).

Differences vs ``perplexity.py``:
  - Loads the model via OGA multimodal processor (vision + embedding + decoder
    sub-sessions), instead of treating it as a text-only LLM.
  - Each sample is one (image, caption) pair. NLL is computed only on the
    caption tokens. The prompt (chat template + image tokens + instruction)
    is the conditioning context and is not scored.
  - Logits mode is probed once, with the same rule as ``perplexity.py``.
    A full-sequence graph prefills prompt+caption in one forward and
    shift-scores the caption positions. A pruned graph emits one position,
    so each caption index gets a fresh Generator and one prefill of
    image + prompt + caption[:j]; the last row scores caption[j].
    Neither path calls ``append_tokens``, ``generate_next_token``, or
    ``rewind_to``.

Plugin EP registration is handled by OGA itself: ``setup_package_env`` exports
``<EP>_EP_PATH`` (e.g. ``MORPHIZEN_EP_PATH``) with the absolute DLL path, which
this subprocess inherits, and OGA's EP Interface self-registers from that env var
on first use. No explicit ``register_execution_provider_library`` call is needed.

Output line that the wrapper parses:
    Perplexity: <float>
"""

import argparse
import ctypes
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


def _preload_runtime_onnxruntime_dll() -> None:
    """Pre-load a custom ``onnxruntime.dll`` BEFORE importing onnxruntime_genai.

    Pip's ``onnxruntime`` may be too old for the active model (e.g. Qwen3.5
    needs ``com.microsoft:CausalConvWithState`` which is only in ORT 1.25+).
    OGA's ``_dll_directory.add_onnxruntime_dependency`` checks ``GetModuleHandleW``
    first and skips its own load if a compatible ``onnxruntime.dll`` is already
    in-process — so pre-loading our package's DLL transparently overrides
    pip's stale one.

    Source order:
      1. ``HIPDNN_PPL_ORT_DLL`` env var (full path) — explicit override
      2. First ``onnxruntime.dll`` found on PATH (package_dir/bin is prepended
         by run_accuracy.py via ``setup_package_env``)

    Silently does nothing on failure — falls back to whatever OGA loads itself,
    which is fine for models that don't need a newer schema.
    """
    explicit = os.environ.get("HIPDNN_PPL_ORT_DLL")
    candidates = []
    if explicit:
        candidates.append(explicit)
    for d in (os.environ.get("PATH", "") or "").split(os.pathsep):
        if not d:
            continue
        cand = os.path.join(d, "onnxruntime.dll")
        if os.path.isfile(cand):
            candidates.append(cand)
            break  # first hit on PATH is enough; OGA's GetModuleHandleW
                   # will then short-circuit its own DLL loading

    for cand in candidates:
        try:
            os.add_dll_directory(os.path.dirname(cand))
            ctypes.WinDLL(cand)
            print(f"[perplexity_vlm] Pre-loaded onnxruntime.dll: {cand}")
            return
        except OSError as e:
            print(f"[perplexity_vlm] WARN: failed to pre-load {cand}: {e}",
                  file=sys.stderr)


_preload_runtime_onnxruntime_dll()

import numpy as np
import onnxruntime_genai as og
import torch
from PIL import Image


# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------

def load_samples(dataset: str, split: str, limit: int) -> List[Tuple[Image.Image, str]]:
    """Load up to ``limit`` (PIL.Image, caption) pairs from a HuggingFace dataset.

    The ``caption`` column may be a string or a list of reference captions; for
    a list the first one is used (flickr30k has 5 per image).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("datasets not installed. Run: pip install datasets") from e

    ds = load_dataset(dataset, split=split, streaming=False)

    samples: List[Tuple[Image.Image, str]] = []
    for ex in ds:
        if len(samples) >= limit:
            break
        img = ex.get("image", None)
        if not isinstance(img, Image.Image):
            continue
        captions = ex.get("caption", None)
        if isinstance(captions, list) and captions:
            cap = str(captions[0]).strip()
        elif isinstance(captions, str):
            cap = captions.strip()
        else:
            continue
        if not cap:
            continue
        samples.append((img.convert("RGB"), cap))

    if not samples:
        raise RuntimeError(f"No usable samples from {dataset}/{split}")
    return samples


# ----------------------------------------------------------------------------
# Prompt building
# ----------------------------------------------------------------------------

def load_chat_template(model_dir: str) -> Optional[str]:
    """Read the chat template from chat_template.jinja or tokenizer_config.json."""
    jinja_path = Path(model_dir) / "chat_template.jinja"
    if jinja_path.is_file():
        with open(jinja_path, "r", encoding="utf-8") as f:
            return f.read()
    tok_cfg_path = Path(model_dir) / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        return tok_cfg.get("chat_template", None)
    return None


def build_messages(instruction: str, with_image: bool = True) -> list:
    """Build a one-turn user message in HF chat-template format.

    ``with_image=False`` is used by ``--mode none``: no image marker is in the
    prompt and the processor is called with ``images=None``, so PPL becomes
    ``P(caption | text-prompt only)``, the language-prior baseline.
    """
    if with_image:
        content = [{"type": "image"}, {"type": "text", "text": instruction}]
    else:
        content = [{"type": "text", "text": instruction}]
    return [{"role": "user", "content": content}]


def pil_to_oga_image(img: Image.Image, image_size: int) -> Tuple[og.Images, str]:
    """Resize PIL image and write to a tempfile so og.Images.open can take it.

    Returns (og.Images, temp_path). Caller should remove temp_path after use.
    """
    im = img.convert("RGB").resize((image_size, image_size))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.close()
    im.save(tmp.name, format="JPEG")
    return og.Images.open(tmp.name), tmp.name


# ----------------------------------------------------------------------------
# Scoring helpers
# ----------------------------------------------------------------------------

@dataclass
class PreparedSample:
    """Tokenised (image, caption) pair.

    ``prompt_inputs`` is the processor bundle for the prompt alone (image
    tensors plus prompt ``input_ids``). ``full_ids`` is prompt + caption, so
    ``full_ids[prompt_len:]`` are the scored caption tokens.
    """
    prompt_inputs: object
    full_ids: np.ndarray
    prompt_len: int

    @property
    def full_len(self) -> int:
        return int(self.full_ids.shape[0])

    @property
    def target_len(self) -> int:
        return self.full_len - self.prompt_len


def _logit_rows(raw) -> np.ndarray:
    rows = np.asarray(raw)
    return rows[0] if rows.ndim == 3 else rows


def _token_nll(row, target_id: int) -> float:
    logits = torch.from_numpy(np.asarray(row, dtype=np.float32)).unsqueeze(0)
    label = torch.tensor([int(target_id)], dtype=torch.int64)
    return float(torch.nn.functional.cross_entropy(logits, label).item())


def _inputs_with_ids(prompt_inputs, token_ids) -> "og.NamedTensors":
    """Same image tensors, with ``input_ids`` replaced by ``token_ids``.

    Caption tokens sit after the image placeholders, so the vision tensors
    from the prompt bundle stay valid. One ``set_inputs`` of this bundle is a
    single prefill of that prefix.
    """
    named = og.NamedTensors()
    ids = np.asarray(token_ids, dtype=np.int32).reshape(1, -1)
    for key in prompt_inputs.keys():
        named[key] = ids if key == "input_ids" else prompt_inputs[key]
    return named


def _pair_image_index(mode: str, i: int, n: int) -> Optional[int]:
    """Index of the image paired with caption ``i``; None means no image.

    The caption is always sample i's own caption; only the image is rotated.
    """
    if mode == "match":
        return i
    if mode == "shuffle":
        # A constant non-zero offset is reproducible and never pairs a caption
        # with its own image (for n >= 2).
        return (i + 1) % n
    if mode == "none":
        return None
    raise ValueError(f"unknown mode {mode!r}")


# ----------------------------------------------------------------------------
# Per-sample PPL computation
# ----------------------------------------------------------------------------

class VlmPplScorer:
    """Teacher-forced caption NLL for one model.

    The graph kind (pruned vs full logits) is fixed per model, so it is probed
    on the first sample and reused.
    """

    def __init__(self, model_dir: str, instruction: str, image_size: int,
                 max_length: int, verbose: bool):
        self.instruction = instruction
        self.image_size = image_size
        self.max_length = max_length
        self.verbose = verbose
        self.model = og.Model(model_dir)
        self.tokenizer = og.Tokenizer(self.model)
        self.processor = self.model.create_multimodal_processor()
        self.template_str = load_chat_template(model_dir)
        self._pruned: Optional[bool] = None
        if verbose:
            print(f"Model loaded; chat_template "
                  f"{'found' if self.template_str else 'missing'}")

    def score(self, img: Optional[Image.Image],
              caption: str) -> Optional[Tuple[float, int]]:
        """Return (sum_nll, num_caption_tokens), or None to skip the sample.

        ``img=None`` runs the text-only prompt (``--mode none``).
        """
        oga_image, tmp_path = None, None
        if img is not None:
            oga_image, tmp_path = pil_to_oga_image(img, self.image_size)
        try:
            sample = self._prepare(oga_image, caption)
            if sample is None:
                return None
            if self._pruned is None:
                self._pruned = self._probe_pruned(sample)
                kind = "PRUNED (last-position only)" if self._pruned else "FULL sequence"
                print(f"[detect] logits mode: {kind}")
            if self.verbose:
                print(f"  [teacher-forced] full_len={sample.full_len}, "
                      f"prompt_len={sample.prompt_len}, "
                      f"target_len={sample.target_len}, "
                      f"path={'pruned' if self._pruned else 'full'}")
            sum_nll = (self._score_pruned if self._pruned else self._score_full)(sample)
            if sum_nll is None:
                if self.verbose:
                    print("  [skip] non-finite NLL")
                return None
            return sum_nll, sample.target_len
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _prepare(self, oga_image, caption: str) -> Optional[PreparedSample]:
        # add_generation_prompt appends the assistant role-open turn, which is
        # where the caption tokens logically continue.
        messages = build_messages(self.instruction, with_image=oga_image is not None)
        template_kwargs = {"add_generation_prompt": True}
        if self.template_str is not None:
            template_kwargs["template_str"] = self.template_str
        prompt_str = self.tokenizer.apply_chat_template(
            json.dumps(messages), **template_kwargs)

        # The processor expands the image placeholder into one token per patch,
        # so prompt_len must be measured from its output.
        prompt_inputs = self.processor(prompt_str, images=oga_image)
        prompt_ids = prompt_inputs["input_ids"].as_numpy().reshape(-1)
        prompt_len = int(prompt_ids.shape[0])

        full_ids = self.processor(
            prompt_str + caption, images=oga_image)["input_ids"].as_numpy().reshape(-1)
        full_len = int(full_ids.shape[0])

        if full_len + 1 > self.max_length:
            if self.verbose:
                print(f"  [skip] full_len={full_len} exceeds max_length="
                      f"{self.max_length}; raise --max_length to evaluate this sample")
            return None

        # The prompt must be a strict prefix of prompt+caption. If BPE merged
        # across the boundary, score from the longest common prefix instead.
        if not np.array_equal(full_ids[:prompt_len], prompt_ids):
            common = 0
            for a, b in zip(full_ids, prompt_ids):
                if a != b:
                    break
                common += 1
            if self.verbose:
                print(f"  [warn] prompt+caption tokenization not strictly "
                      f"prefix-extending (common={common} of {prompt_len}); "
                      f"using common as boundary")
            prompt_len = common

        if full_len - prompt_len <= 0:
            if self.verbose:
                print(f"  [skip] caption tokenized to 0 new tokens "
                      f"(prompt_len={prompt_len}, full_len={full_len})")
            return None
        return PreparedSample(prompt_inputs, full_ids, prompt_len)

    def _generator(self, full_len: int):
        params = og.GeneratorParams(self.model)
        # The search horizon in genai_config.json can be shorter than the image
        # prompt. It only sizes the KV buffer; scoring never samples.
        params.set_search_options(
            do_sample=False,
            top_k=1,
            max_length=max(self.max_length, full_len + 1),
            min_length=full_len,
        )
        return og.Generator(self.model, params)

    def _probe_pruned(self, sample: PreparedSample) -> bool:
        """True when a multi-token prefill still returns a single logit row.

        Same test as ``perplexity.detect_pruned_logits``: the prune-logits gather
        keeps only the last position, so the sequence dim is 1 instead of the
        input length.
        """
        n_in = int(sample.prompt_inputs["input_ids"].as_numpy().size)
        generator = self._generator(n_in)
        try:
            generator.set_inputs(sample.prompt_inputs)
            rows = _logit_rows(generator.get_output("logits"))
        finally:
            del generator
        return int(rows.shape[0]) < n_in

    def _score_pruned(self, sample: PreparedSample) -> Optional[float]:
        """One prefill per caption position: image + prompt + caption[:j]."""
        sum_nll = 0.0
        for j in range(sample.target_len):
            generator = self._generator(sample.full_len)
            try:
                generator.set_inputs(_inputs_with_ids(
                    sample.prompt_inputs, sample.full_ids[:sample.prompt_len + j]))
                row = _logit_rows(generator.get_output("logits"))[-1]
            finally:
                del generator
            nll = _token_nll(row, sample.full_ids[sample.prompt_len + j])
            if not np.isfinite(nll):
                return None
            sum_nll += nll
        return sum_nll

    def _score_full(self, sample: PreparedSample) -> Optional[float]:
        """One prefill of prompt+caption; row t predicts token t+1."""
        generator = self._generator(sample.full_len)
        try:
            generator.set_inputs(_inputs_with_ids(sample.prompt_inputs, sample.full_ids))
            rows = _logit_rows(generator.get_output("logits"))
        finally:
            del generator
        sum_nll = 0.0
        for j in range(sample.target_len):
            nll = _token_nll(rows[sample.prompt_len + j - 1],
                             sample.full_ids[sample.prompt_len + j])
            if not np.isfinite(nll):
                return None
            sum_nll += nll
        return sum_nll


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(args):
    print(f"Calculating VLM Perplexity on dataset={args.dataset} "
          f"split={args.split} limit={args.limit} mode={args.mode} ...")

    if args.verbose:
        print("Loading dataset ...")
    samples = load_samples(args.dataset, args.split, args.limit)
    if args.verbose:
        print(f"Loaded {len(samples)} samples")
        print("Loading model (this may JIT-compile EPs on first call) ...")

    scorer = VlmPplScorer(args.model, args.instruction, args.image_size,
                          args.max_length, args.verbose)

    total_nll = 0.0
    total_tokens = 0
    skipped = 0
    per_sample_ppl: List[float] = []

    n = len(samples)
    for i in range(n):
        caption = samples[i][1]
        if args.verbose:
            print(f"[{i + 1}/{n}] caption={caption[:60]!r}")
        image_idx = _pair_image_index(args.mode, i, n)
        img = None if image_idx is None else samples[image_idx][0]
        result = scorer.score(img, caption)
        if result is None:
            skipped += 1
            continue
        sum_nll, n_target = result
        total_nll += sum_nll
        total_tokens += n_target
        sample_ppl = float(np.exp(sum_nll / n_target))
        per_sample_ppl.append(sample_ppl)
        if args.verbose:
            print(f"  sum_nll={sum_nll:.4f} n_target={n_target} "
                  f"sample_ppl={sample_ppl:.4f}")

    if total_tokens == 0:
        print("ERROR: every sample was skipped; cannot compute PPL", file=sys.stderr)
        sys.exit(2)

    avg_nll = total_nll / total_tokens
    ppl = float(np.exp(avg_nll))

    # The summary block doubles as the parser target for the wrapper.
    print("=" * 60)
    print(f"Samples evaluated : {n - skipped}/{n}")
    print(f"Skipped           : {skipped}")
    print(f"Total target tokens: {total_tokens}")
    print(f"Average NLL       : {avg_nll:.6f}")
    per = np.array(per_sample_ppl, dtype=np.float64)
    print(f"Per-sample PPL    : "
          f"min={per.min():.4f} mean={per.mean():.4f} "
          f"median={np.median(per):.4f} max={per.max():.4f}")
    # IMPORTANT: `Perplexity: <value>` is the line tests/ppl.py /
    # tests/ppl_vlm.py regex-parses. Keep this format stable.
    print(f"Perplexity: {ppl:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conditional VLM perplexity over an image+caption dataset",
    )
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="OGA model dir (must contain genai_config.json + sub-models)")
    parser.add_argument("-d", "--dataset", type=str, default="lmms-lab/flickr30k",
                        help="HF dataset name. Default: lmms-lab/flickr30k")
    parser.add_argument("-s", "--split", type=str, default="test",
                        help="HF split. Default: test")
    parser.add_argument("-n", "--limit", type=int, default=50,
                        help="Number of samples to evaluate. Default: 50")
    parser.add_argument("--instruction", type=str,
                        default="Describe this image briefly.",
                        help="Text instruction sent with the image")
    parser.add_argument("--image_size", type=int, default=896,
                        help="Square resize size for input images. Default: 896")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="OGA search.max_length used for KV buffer sizing. "
                             "Default 1024 (fits image_size=896, whose prompt is "
                             "~800 image tokens); raise for larger images or "
                             "longer captions.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="match",
                        choices=["match", "shuffle", "none"],
                        help="Visual-grounding mode. 'match': caption_i paired "
                             "with its own image_i (canonical PPL). 'shuffle': "
                             "caption_i paired with image_(i+1)%%N (visual-"
                             "neglect probe). 'none': caption_i with no image "
                             "(language-prior baseline). Run all three to "
                             "compute PPL_shuf/PPL_match and PPL_none/PPL_match "
                             "ratios - both should be > 1 for a healthy VLM.")
    # accept (and ignore) --device for symmetry with perplexity.py callers
    parser.add_argument("--device", type=str, required=False, default="cpu",
                        choices=["cpu", "aie", "gpu"], help="(ignored)")
    main(parser.parse_args())
