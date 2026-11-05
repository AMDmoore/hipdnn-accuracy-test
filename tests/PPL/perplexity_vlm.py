"""VLM (Vision-Language Model) Perplexity runner.

Computes conditional perplexity PPL(caption | image) over a small image+caption
dataset (default: HuggingFace ``lmms-lab/flickr30k`` test split).

Differences vs ``perplexity.py``:
  - Loads the model via OGA multimodal processor (vision + embedding + decoder
    sub-sessions), instead of treating it as a text-only LLM.
  - Each sample is one (image, caption) pair. NLL is computed only on the
    caption tokens. The prompt (chat template + image tokens + instruction)
    is the conditioning context and is not scored.
  - The caption is scored in the answer slot: the chat template is rendered
    with thinking turned off, so the first caption token is predicted exactly
    where the model would start its reply. The prompt and the caption are
    tokenized separately and joined at the id level, which reproduces the
    context the model sees when it generates (a joint tokenization can merge
    the template's trailing newlines with the caption's first token).

Scoring (default, "sampled"): a deterministic set of caption positions is
scored per sample, with one Generator per sample. The first position (``p0``)
is predicted from the prompt-only prefill. The rest of the caption, positions
[1, L), is cut into ``--segments`` equal segments and the midpoint of each is
scored (``seg1`` .. ``segK``). The generator is advanced to each point with
``append_tokens``; every append carries at least two tokens, so it runs the
same multi-token kernels as a prefill rather than the single-token decode
kernels. If a caption is too short for that, segments are dropped (down to
``p0`` alone). The positions depend only on the caption length, so every run
of the same dataset slice scores the same tokens.

``--exact`` scores every caption token instead. A pruned-logits graph emits one
row per forward, so each caption index gets a fresh Generator and one prefill;
a full-logits graph scores all positions from a single prefill.

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

def load_samples(dataset: str, split: str,
                 nsamples: float) -> List[Tuple[Image.Image, str]]:
    """Load the first ``int(nsamples * len(split))`` (PIL.Image, caption) pairs.

    ``nsamples`` is a fraction of the split, as in ``perplexity.py``. Taking a
    prefix of the split keeps the evaluated slice identical across runs.

    The ``caption`` column may be a string or a list of reference captions; for
    a list the first one is used (flickr30k has 5 per image).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("datasets not installed. Run: pip install datasets") from e

    ds = load_dataset(dataset, split=split, streaming=False)

    limit = int(nsamples * len(ds))
    if limit < 1:
        raise ValueError(
            f"nsamples={nsamples} selects {limit} of {len(ds)} samples; "
            f"increase nsamples")

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

    @property
    def caption_ids(self) -> np.ndarray:
        return self.full_ids[self.prompt_len:]


# (stage label, NLL of one scored caption token)
Scored = List[Tuple[str, float]]

EXACT_STAGE = "tok"


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


def sample_positions(n_tokens: int, segments: int) -> List[int]:
    """Caption positions scored by the sampled scheme.

    Position ``j`` means: context = prompt + caption[:j], target = caption[j].
    The result starts at 0 (the prompt-only prefill), followed by the midpoint
    of each of ``segments`` equal segments of [1, n_tokens). Consecutive
    positions differ by at least 2 so every append is a multi-token step; when
    the caption is too short for that, segments are dropped one at a time.
    """
    for k in range(segments, 0, -1):
        pos = [0] + [1 + ((2 * i - 1) * (n_tokens - 1)) // (2 * k)
                     for i in range(1, k + 1)]
        if all(b - a >= 2 for a, b in zip(pos, pos[1:])):
            return pos
    return [0]


def stage_label(stage: int) -> str:
    return "p0" if stage == 0 else f"seg{stage}"


# ----------------------------------------------------------------------------
# Per-sample PPL computation
# ----------------------------------------------------------------------------

class VlmPplScorer:
    """Teacher-forced caption NLL for one model."""

    def __init__(self, model_dir: str, instruction: str, image_size: int,
                 max_length: int, verbose: bool, exact: bool = False,
                 segments: int = 2):
        self.instruction = instruction
        self.image_size = image_size
        self.max_length = max_length
        self.verbose = verbose
        self.exact = exact
        self.segments = segments
        self.model = og.Model(model_dir)
        self.tokenizer = og.Tokenizer(self.model)
        self.processor = self.model.create_multimodal_processor()
        self.template_str = load_chat_template(model_dir)
        # With thinking on, the reply starts with a reasoning block, so a
        # caption would be scored against the wrong slot. Templates that do not
        # read the variable ignore it.
        try:
            self.tokenizer.update_options(
                chat_template_kwargs=json.dumps({"enable_thinking": False}))
        except Exception as e:  # tokenizer without chat_template_kwargs support
            print(f"[warn] cannot disable thinking via chat_template_kwargs: {e}",
                  file=sys.stderr)
        self._pruned: Optional[bool] = None
        if verbose:
            print(f"Model loaded; chat_template "
                  f"{'found' if self.template_str else 'missing'}")

    def score(self, img: Optional[Image.Image], caption: str) -> Optional[Scored]:
        """Return one (stage, NLL) per scored caption token, or None to skip.

        ``img=None`` runs the text-only prompt (``--mode none``).
        """
        oga_image, tmp_path = None, None
        if img is not None:
            oga_image, tmp_path = pil_to_oga_image(img, self.image_size)
        try:
            sample = self._prepare(oga_image, caption)
            if sample is None:
                return None
            if self.verbose:
                print(f"  [teacher-forced] full_len={sample.full_len}, "
                      f"prompt_len={sample.prompt_len}, "
                      f"target_len={sample.target_len}")
            if self.exact:
                scored = self._score_exact(sample)
            else:
                scored = self._score_sampled(sample)
            if scored is None or not all(np.isfinite(v) for _, v in scored):
                if self.verbose:
                    print("  [skip] non-finite NLL")
                return None
            return scored
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
        prompt_ids = prompt_inputs["input_ids"].as_numpy().reshape(-1).astype(np.int32)

        # Tokenizing the caption on its own keeps the prompt a strict prefix of
        # the sequence: encoding prompt+caption together can merge the
        # template's trailing newlines with the caption's first token, which
        # is not the context the model has when it generates.
        caption_ids = np.asarray(self.tokenizer.encode(caption), dtype=np.int32).reshape(-1)
        if caption_ids.size == 0:
            if self.verbose:
                print("  [skip] caption tokenized to 0 tokens")
            return None

        full_ids = np.concatenate([prompt_ids, caption_ids])
        if int(full_ids.shape[0]) + 1 > self.max_length:
            if self.verbose:
                print(f"  [skip] full_len={full_ids.shape[0]} exceeds max_length="
                      f"{self.max_length}; raise --max_length to evaluate this sample")
            return None
        return PreparedSample(prompt_inputs, full_ids, int(prompt_ids.shape[0]))

    def _generator(self, full_len: int, whole_prefill: bool = False):
        params = og.GeneratorParams(self.model)
        if whole_prefill:
            # Chunked prefill keeps only the last chunk's logits rows, so a
            # caller that reads every row needs the prompt in one forward.
            params.set_search_options(chunk_size=0)
        # The search horizon in genai_config.json can be shorter than the image
        # prompt. It only sizes the KV buffer; scoring never samples.
        params.set_search_options(
            do_sample=False,
            top_k=1,
            max_length=max(self.max_length, full_len + 1),
            min_length=full_len,
        )
        return og.Generator(self.model, params)

    # -- sampled scheme ------------------------------------------------------

    def _score_sampled(self, sample: PreparedSample) -> Optional[Scored]:
        positions = sample_positions(sample.target_len, self.segments)
        if self.verbose:
            print(f"  [sampled] positions={positions} of {sample.target_len}")
        try:
            return self._score_positions_continuous(sample, positions)
        except RuntimeError as e:
            raise RuntimeError(
                f"sampled scoring failed ({e}); it needs append_tokens "
                f"(continuous decoding). Use --exact for a path without it.") from e

    def _score_positions_continuous(self, sample: PreparedSample,
                                    positions: List[int]) -> Scored:
        """One Generator: prefill the prompt, then append up to each position."""
        caption = sample.caption_ids
        scored: Scored = []
        generator = self._generator(sample.full_len)
        try:
            generator.set_inputs(sample.prompt_inputs)
            done = 0
            for stage, pos in enumerate(positions):
                if pos > done:
                    generator.append_tokens(np.ascontiguousarray(caption[done:pos]))
                    done = pos
                row = _logit_rows(generator.get_output("logits"))[-1]
                scored.append((stage_label(stage), _token_nll(row, caption[pos])))
        finally:
            del generator
        return scored

    def _score_positions_fresh(self, sample: PreparedSample, positions) -> Scored:
        """One prefill of prompt + caption[:pos] per position."""
        caption = sample.caption_ids
        scored: Scored = []
        for pos in positions:
            generator = self._generator(sample.full_len)
            try:
                generator.set_inputs(_inputs_with_ids(
                    sample.prompt_inputs, sample.full_ids[:sample.prompt_len + pos]))
                row = _logit_rows(generator.get_output("logits"))[-1]
            finally:
                del generator
            scored.append((EXACT_STAGE, _token_nll(row, caption[pos])))
        return scored

    # -- exact scheme (every caption token) ----------------------------------

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

    def _score_exact(self, sample: PreparedSample) -> Optional[Scored]:
        if self._pruned is None:
            self._pruned = self._probe_pruned(sample)
            kind = "PRUNED (last-position only)" if self._pruned else "FULL sequence"
            print(f"[detect] logits mode: {kind}")
        if self._pruned:
            return self._score_exact_pruned(sample)
        return self._score_exact_full(sample)

    def _score_exact_pruned(self, sample: PreparedSample) -> Scored:
        """One prefill per caption position: image + prompt + caption[:j]."""
        return self._score_positions_fresh(sample, range(sample.target_len))

    def _score_exact_full(self, sample: PreparedSample) -> Scored:
        """One prefill of prompt+caption; row t predicts token t+1."""
        generator = self._generator(sample.full_len, whole_prefill=True)
        try:
            generator.set_inputs(_inputs_with_ids(sample.prompt_inputs, sample.full_ids))
            rows = _logit_rows(generator.get_output("logits"))
        finally:
            del generator
        if rows.shape[0] < sample.full_len:
            raise RuntimeError(
                f"full-logits scoring needs {sample.full_len} logits rows but the "
                f"model returned {rows.shape[0]}")
        caption = sample.caption_ids
        return [(EXACT_STAGE, _token_nll(rows[sample.prompt_len + j - 1], caption[j]))
                for j in range(sample.target_len)]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(args):
    print(f"Calculating VLM Perplexity on dataset={args.dataset} "
          f"split={args.split} nsamples={args.nsamples} mode={args.mode} "
          f"scoring={'exact' if args.exact else f'sampled(segments={args.segments})'} ...")

    if not 0.0 < args.nsamples <= 1.0:
        print("ERROR: nsamples must be a fraction in (0, 1]", file=sys.stderr)
        sys.exit(2)

    if args.verbose:
        print("Loading dataset ...")
    samples = load_samples(args.dataset, args.split, args.nsamples)
    if args.verbose:
        print(f"Loaded {len(samples)} samples")
        print("Loading model (this may JIT-compile EPs on first call) ...")

    scorer = VlmPplScorer(args.model, args.instruction, args.image_size,
                          args.max_length, args.verbose, exact=args.exact,
                          segments=args.segments)

    total_nll = 0.0
    total_points = 0
    skipped = 0
    per_sample_ppl: List[float] = []
    stage_totals: dict = {}   # stage label -> [sum NLL, count], first-seen order

    n = len(samples)
    for i in range(n):
        caption = samples[i][1]
        if args.verbose:
            print(f"[{i + 1}/{n}] caption={caption[:60]!r}")
        image_idx = _pair_image_index(args.mode, i, n)
        img = None if image_idx is None else samples[image_idx][0]
        scored = scorer.score(img, caption)
        if scored is None:
            skipped += 1
            continue
        nlls = [v for _, v in scored]
        for label, v in scored:
            acc = stage_totals.setdefault(label, [0.0, 0])
            acc[0] += v
            acc[1] += 1
        total_nll += sum(nlls)
        total_points += len(nlls)
        sample_ppl = float(np.exp(np.mean(nlls)))
        per_sample_ppl.append(sample_ppl)
        if args.verbose:
            print(f"  scored={len(nlls)} sample_ppl={sample_ppl:.4f}")

    if total_points == 0:
        print("ERROR: every sample was skipped; cannot compute PPL", file=sys.stderr)
        sys.exit(2)

    # Every scored point carries equal weight in the aggregate.
    avg_nll = total_nll / total_points
    ppl = float(np.exp(avg_nll))

    # The summary block doubles as the parser target for the wrapper.
    print("=" * 60)
    print(f"Samples evaluated : {n - skipped}/{n}")
    print(f"Skipped           : {skipped}")
    print(f"Scored points     : {total_points}")
    print(f"Average NLL       : {avg_nll:.6f}")
    if not args.exact:
        print("Stage NLL         : " + " ".join(
            f"{label}={s / c:.6f}(n={c})" for label, (s, c) in stage_totals.items()))
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
    parser.add_argument("-n", "--nsamples", type=float, default=0.002,
                        help="Fraction of the split to evaluate, taken from its "
                             "start (e.g. 0.002 = 0.2%%). Default: 0.002")
    parser.add_argument("--segments", type=int, default=2,
                        help="Sampled scoring: number of caption segments whose "
                             "midpoints are scored in addition to the first "
                             "caption token. Default: 2")
    parser.add_argument("--exact", action="store_true", default=False,
                        help="Score every caption token instead of the sampled "
                             "positions (slow: one prefill per token on a "
                             "pruned-logits graph)")
    parser.add_argument("--instruction", type=str,
                        default="Describe this image briefly without any introductory phrase.",
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
