"""VLM (Vision-Language Model) Perplexity runner.

Computes conditional perplexity PPL(caption | image) over a small image+caption
dataset (default: HuggingFace ``lmms-lab/flickr30k`` test split).

Differences vs ``perplexity.py``:
  - Loads the model via OGA multimodal processor (vision + embedding + decoder
    sub-sessions), instead of treating it as a text-only LLM.
  - Each sample is one (image, caption) pair, sent through the full multimodal
    pipeline via ``generator.set_inputs(processor(prompt, images=...))`` —
    NOT ``append_tokens`` (which is text-only).
  - NLL is computed only on the *caption* tokens. The prompt portion (chat
    template + image tokens + instruction) is masked with ``-100`` in the
    label tensor so it does not contribute to PPL.

Plugin EP registration (MorphiZenEP) is handled centrally by
``tests/_ep_bootstrap.py`` when this script is invoked through the orchestrator.
Running directly from a shell skips that wrapper, so the active EP is whatever
``genai_config.json`` selects.

Output line that the wrapper parses:
    Perplexity: <float>
"""

import argparse
import ctypes
import json
import os
import sys
import tempfile
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


def _register_plugin_eps_self() -> None:
    """Register plugin EPs (e.g. MorphiZenEP) found on PATH.

    Mirrors what tests/_ep_bootstrap.py does centrally for the LLM tests, but
    runs HERE (after our ORT pre-load) so the multimodal sub-sessions in
    genai_config can resolve `provider_options: [{ "MorphiZenEP": {} }]`.

    PPLVLMTest.execute() bypasses ``_ep_bootstrap.py`` injection — the bootstrap
    imports OGA before our ORT pre-load can run, defeating the workaround.
    """
    if not hasattr(og, "register_execution_provider_library"):
        return
    plugin_eps = {"MorphiZenEP": "onnxruntime_morphizen_ep.dll"}
    for ep_name, dll_name in plugin_eps.items():
        for d in (os.environ.get("PATH", "") or "").split(os.pathsep):
            if not d:
                continue
            cand = os.path.join(d, dll_name)
            if os.path.isfile(cand):
                try:
                    og.register_execution_provider_library(ep_name, cand)
                    print(f"[perplexity_vlm] Registered plugin EP: "
                          f"{ep_name} -> {cand}")
                except Exception as e:
                    print(f"[perplexity_vlm] WARN: failed to register "
                          f"{ep_name}: {e}", file=sys.stderr)
                break


_register_plugin_eps_self()


# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------

def load_flickr30k_test(limit: int) -> List[Tuple[Image.Image, str]]:
    """Load (image, caption) pairs from HuggingFace ``lmms-lab/flickr30k``.

    Returns the first ``limit`` samples as a list of (PIL.Image, caption_str).
    Each sample's caption is the first of the 5 reference captions.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "datasets not installed. Run: pip install datasets"
        ) from e

    ds = load_dataset("lmms-lab/flickr30k", split="test", streaming=False)

    samples: List[Tuple[Image.Image, str]] = []
    for ex in ds:
        if len(samples) >= limit:
            break
        img = ex.get("image", None)
        if img is None or not isinstance(img, Image.Image):
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
        raise RuntimeError("Loaded zero usable samples from lmms-lab/flickr30k")
    return samples


# ----------------------------------------------------------------------------
# Prompt building (mirrors mmmu_eval.py for chat-template handling)
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

    ``with_image=True``  -> [image, text] content (the canonical multimodal path).
    ``with_image=False`` -> text-only content. Used for the visual-neglect
    sanity mode ``--mode none``: no image marker is present in the prompt, the
    chat template emits no ``<|vision_start|>...<|vision_end|>`` block, and
    the OGA processor is called with ``images=None`` so the vision encoder
    is bypassed entirely. PPL in this mode = ``P(caption | text-prompt only)``,
    i.e. the model's pure language-prior caption likelihood.
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
# Per-sample PPL computation
# ----------------------------------------------------------------------------

def compute_sample_nll(
    model: og.Model,
    tokenizer: og.Tokenizer,
    processor,
    template_str: Optional[str],
    img: Optional[Image.Image],
    caption: str,
    instruction: str,
    image_size: int,
    max_length: int,
    verbose: bool,
    topk: int = 0,
) -> Optional[Tuple[float, int, dict]]:
    """Run one (image, caption) sample. Returns (sum_nll, num_target_tokens).

    ``img`` semantics control which of the three visual-neglect modes runs:
      - ``img is not None`` -> standard image-conditioned path (modes ``match``
        and ``shuffle``; the caller decides which image is paired with which
        caption).
      - ``img is None``     -> text-only path. ``messages`` are built without
        an image marker, ``processor(...)`` is called with ``images=None``,
        and the vision encoder + image-feature scatter are bypassed entirely.

    Returns None on a recoverable error (per-sample failure should not abort
    the whole run; we just skip the sample and continue).
    """
    text_only = img is None
    if not text_only:
        oga_image, tmp_path = pil_to_oga_image(img, image_size)
    else:
        oga_image = None
        tmp_path = None

    try:
        # Build prompt with (or without) image placeholder + instruction.
        # add_generation_prompt appends the assistant role-open turn so the
        # model is "ready to answer" — that's where the caption tokens will
        # logically be appended.
        messages = build_messages(instruction, with_image=not text_only)
        if template_str is not None:
            prompt_str = tokenizer.apply_chat_template(
                json.dumps(messages),
                template_str=template_str,
                add_generation_prompt=True,
            )
        else:
            prompt_str = tokenizer.apply_chat_template(
                json.dumps(messages),
                add_generation_prompt=True,
            )

        # Dry run: tokenize prompt-only (with the image, if present) to learn
        # how many tokens the prompt actually occupies. With an image, the
        # processor expands the ``<|image_pad|>`` placeholder into N image
        # tokens (N depends on ``image_grid_thw``); without one, the prompt is
        # plain text. Either way ``prompt_len`` is the boundary that label-mask
        # math depends on, and it MUST be measured per-mode rather than reused
        # across modes — the text-only prompt is shorter by ~768 tokens.
        # OGA NamedTensors values are wrapped in ``onnxruntime_genai.Tensor``;
        # use ``.as_numpy()`` to get the underlying ndarray.
        prompt_inputs = processor(prompt_str, images=oga_image)
        prompt_ids = prompt_inputs["input_ids"].as_numpy().reshape(-1)
        prompt_len = int(prompt_ids.shape[0])

        # Full sequence: prompt + caption. The processor will tokenize the
        # caption as plain text appended after the assistant role-open marker.
        full_str = prompt_str + caption
        full_inputs = processor(full_str, images=oga_image)
        full_ids = full_inputs["input_ids"].as_numpy().reshape(-1)
        full_len = int(full_ids.shape[0])

        target_len = full_len - prompt_len
        if target_len <= 0:
            if verbose:
                print(f"  [skip] caption tokenized to 0 new tokens "
                      f"(prompt_len={prompt_len}, full_len={full_len})")
            return None
        if full_len + 1 > max_length:
            if verbose:
                print(f"  [skip] full_len={full_len} exceeds max_length="
                      f"{max_length}; raise --max_length to evaluate this sample")
            return None

        # Sanity: prompt_ids must be a strict prefix of full_ids. If not, the
        # caption text changed how the prompt was tokenised (BPE merges across
        # the boundary). Fall back to the longest common prefix.
        if not np.array_equal(full_ids[:prompt_len], prompt_ids):
            min_len = min(prompt_len, full_len)
            common = 0
            for k in range(min_len):
                if full_ids[k] != prompt_ids[k]:
                    break
                common += 1
            if verbose:
                print(f"  [warn] prompt+caption tokenization not strictly "
                      f"prefix-extending (common={common} of {prompt_len}); "
                      f"using common as boundary")
            prompt_len = common
            target_len = full_len - prompt_len
            if target_len <= 0:
                return None

        # Run prefill via the multimodal generator. set_inputs binds image
        # features + token ids; generate_next_token triggers the prefill
        # compute (and a one-step decode whose sampling we ignore — we only
        # want the prefill logits cached behind get_output("logits")).
        params = og.GeneratorParams(model)
        # Force the search horizon to fit our sequence; the value baked in
        # genai_config.json (e.g. 384) may be too tight for longer captions.
        params.set_search_options(
            do_sample=False,
            top_k=1,
            max_length=max(max_length, full_len + 1),
            min_length=full_len,  # prevent early stop before prefill finishes
        )

        # We don't yet know if the model's text decoder graph emits all-position
        # logits ([B, T, V]) or only the last-position ([B, 1, V]) — the former
        # is the canonical HF export; the latter is the DML-optimized export
        # that pre-applies the "select last token" inside the graph for cheaper
        # autoregressive decode. Try the all-position path first; fall back to
        # a teacher-forced per-token decode loop if logits seq dim is short.
        # ``full_inputs`` (set on prompt_inputs) is the prompt-only multimodal
        # input — its image features only need to be bound once; subsequent
        # decode steps reuse the KV cache.
        generator = og.Generator(model, params)
        sum_nll: float
        try:
            # Bind image features + prompt token ids. Do NOT use ``full_inputs``
            # (which contains the caption appended to the prompt) because we
            # need to inject caption tokens one at a time in the per-token
            # path — and the all-position path doesn't care whether the caption
            # was bound at set_inputs time or appended via append_tokens, since
            # in both cases the graph runs the same prefill compute.
            generator.set_inputs(prompt_inputs)
            # Append the caption tokens so the prefill covers full_len. For
            # the all-position path this is equivalent to set_inputs(full_inputs);
            # for the last-position path we'll rewind + replay below.
            caption_ids = full_ids[prompt_len:].astype(np.int32).tolist()
            generator.append_tokens(caption_ids)
            generator.generate_next_token()  # triggers prefill

            logits_raw = generator.get_output("logits")
            logits_all = np.asarray(logits_raw)
            if logits_all.ndim == 3:
                logits_all = logits_all[0]
            if logits_all.ndim != 2:
                if verbose:
                    print(f"  [skip] unexpected logits shape "
                          f"{logits_all.shape}; expected [T,V]")
                return None

            if logits_all.shape[0] >= full_len:
                # All-position-logits path (canonical HF export). Standard
                # shift: logits[t] predicts token at position t+1. Loss is
                # computed only on positions where the target is a CAPTION
                # token, i.e. shift_labels[i] = full_ids[i+1] with i in
                # [prompt_len-1, full_len-1).
                logits_slice = logits_all[:full_len].astype(np.float32)

                if verbose:
                    target_logits = logits_slice[prompt_len - 1: full_len - 1]
                    n_bad = int(np.sum(~np.isfinite(target_logits)))
                    if n_bad > 0:
                        print(f"  [diag] {n_bad} non-finite logit values in "
                              f"target window (shape={target_logits.shape}); "
                              f"this sample will yield NaN PPL")

                # Per-target-token NLL + top-1 prediction. The row that
                # predicts target[j] = full_ids[prompt_len + j] is row
                # (prompt_len - 1 + j) of the all-position logits. Computing
                # per token (rather than one masked sum) is mathematically
                # identical for the aggregate NLL, but additionally lets the
                # caller dump a per-position EP-vs-CPU comparison (top-1
                # agreement + per-token NLL delta) — a finer correctness
                # signal than aggregate PPL alone.
                tgt_rows = logits_slice[prompt_len - 1: full_len - 1]      # [target_len, V]
                tgt_ids_np = full_ids[prompt_len:full_len].astype(np.int64)  # [target_len]
                lt = torch.from_numpy(tgt_rows)
                lbl = torch.from_numpy(tgt_ids_np)
                per_tok = torch.nn.functional.cross_entropy(
                    lt, lbl, reduction="none")
                sum_nll = float(per_tok.sum().item())
                tgt_ids_out = tgt_ids_np.tolist()
                per_tok_out = [float(x) for x in per_tok.tolist()]
                top1_out = [int(x) for x in np.argmax(tgt_rows, axis=1).tolist()]
                # Optional top-k / logit decomposition for the flip / max-dev
                # investigation. NLL = lse - target_logit, so dumping both lets
                # the comparison split the EP-vs-CPU NLL delta into a logit-
                # value component and a normalization (logsumexp) component.
                target_logit_out, lse_out, topk_ids_out, topk_logits_out = [], [], [], []
                if topk and topk > 0:
                    lse = torch.logsumexp(lt, dim=-1)
                    tlog = lt.gather(1, lbl.view(-1, 1)).squeeze(1)
                    kk = min(topk, lt.shape[1])
                    tvals, tids = torch.topk(lt, kk, dim=-1)
                    lse_out = [float(x) for x in lse.tolist()]
                    target_logit_out = [float(x) for x in tlog.tolist()]
                    topk_ids_out = [[int(x) for x in row] for row in tids.tolist()]
                    topk_logits_out = [[float(x) for x in row] for row in tvals.tolist()]
            else:
                # Last-position-only path. The graph emits logits of shape
                # [B, 1, V] = P(next | full prefix). For PPL we need
                #   NLL = -sum_{t=prompt_len-1..full_len-2} log P(full_ids[t+1] | full_ids[:t+1])
                # which is target_len contributions. We get them via
                # teacher-forced per-token decode:
                #   - the just-completed prefill of full_ids[:full_len] gave us
                #     logits = P(? | full_ids[:full_len]) (one position past the
                #     end — useful only as a sanity check, not a target).
                #   - rewind to position prompt_len, then for i in 0..target_len-1:
                #       * append_tokens([full_ids[prompt_len + i]]) — feeds the
                #         actual target token (teacher forcing) and runs one
                #         forward pass; logits_after = P(? | prefix + target[..i])
                #         is the prediction for target[i+1], NOT target[i].
                #     So the logits we want for target[i] come from the forward
                #     pass that consumes target[i-1] (or the prefill of the
                #     prompt for target[0]).
                # Implementation: rewind to prompt_len-1, then loop:
                #   step k=0: prefix = full_ids[:prompt_len], read logits =
                #     P(? | prompt) -> NLL for full_ids[prompt_len] = target[0]
                #   step k>0: append target[k-1], read logits = P(? | prompt+target[:k])
                #     -> NLL for target[k]
                # ``append_tokens`` queues tokens and ``generate_next_token``
                # runs the forward pass + samples (we ignore the sample).
                if verbose:
                    print(f"  [last-pos-only] full_len={full_len}, "
                          f"prompt_len={prompt_len}, target_len={target_len}; "
                          f"running per-token decode (this is the slow path)")

                # Rewind to position prompt_len. token_count() reports the
                # current sequence length; after the prefill of full_len + 1
                # sampled tokens it is full_len + 1. The exact value doesn't
                # matter — rewind_to(prompt_len) puts the KV cache back to the
                # state right after the prompt was consumed.
                generator.rewind_to(prompt_len)

                # Step k=0: read logits at position prompt_len-1, predicting
                # target[0]. The KV cache currently contains exactly prompt_len
                # tokens; the LAST forward pass that filled position prompt_len-1
                # was the prompt prefill we just rewound past. ``get_output``
                # still returns the cached logits from that prefill — but to
                # be safe we re-run a one-step "no-op" by appending a single
                # token, reading the logits, then rewinding again.
                # Actually: after rewind_to(prompt_len), the next forward pass
                # is decode-mode. Append target[0] to consume it, read logits
                # — those logits are P(? | prompt + target[0]) = prediction for
                # target[1], not target[0]. That doesn't help.
                # Solution: rewind to prompt_len - 1 and append the LAST prompt
                # token. The forward pass on that single token, with KV cache
                # holding prompt_len-1 tokens, computes logits for position
                # prompt_len-1 = P(? | prompt[:prompt_len]) = target[0].
                generator.rewind_to(prompt_len - 1)
                last_prompt_tok = int(full_ids[prompt_len - 1])
                generator.append_tokens([last_prompt_tok])
                generator.generate_next_token()
                logits_step = np.asarray(generator.get_output("logits"))
                if logits_step.ndim == 3:
                    logits_step = logits_step[0]
                # logits_step shape is [1, V] (last position of the 1-token
                # decode step).
                logits_t = torch.from_numpy(logits_step.astype(np.float32))
                loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                sum_nll = 0.0
                # Per-token records (parallel to the all-position path) so the
                # caller can dump an EP-vs-CPU top-1 / NLL comparison.
                tgt_ids_out = []
                per_tok_out = []
                top1_out = []
                target_logit_out, lse_out, topk_ids_out, topk_logits_out = [], [], [], []

                def _record_step(row_t, tid, snll):
                    # row_t: torch tensor [V] of logits for this decode step.
                    tgt_ids_out.append(int(tid))
                    per_tok_out.append(float(snll))
                    top1_out.append(int(row_t.argmax().item()))
                    if topk and topk > 0:
                        lse_out.append(float(torch.logsumexp(row_t, dim=-1).item()))
                        target_logit_out.append(float(row_t[tid].item()))
                        kk = min(topk, row_t.shape[0])
                        tv, ti = torch.topk(row_t, kk)
                        topk_ids_out.append([int(x) for x in ti.tolist()])
                        topk_logits_out.append([float(x) for x in tv.tolist()])

                target_id = int(full_ids[prompt_len])
                step_logits = logits_t[-1:, :]
                step_label = torch.tensor([target_id], dtype=torch.int64)
                step_nll = loss_fct(step_logits, step_label).item()
                if not np.isfinite(step_nll):
                    if verbose:
                        print(f"  [diag] non-finite NLL at decode step 0 "
                              f"(target_id={target_id}); aborting sample")
                    return None
                sum_nll += step_nll
                _record_step(step_logits[-1], target_id, step_nll)

                # Steps k=1..target_len-1: append target[k-1], decode 1 step,
                # logits = P(? | prompt + target[:k]) = prediction for target[k].
                for k in range(1, target_len):
                    prev_target = int(full_ids[prompt_len + k - 1])
                    target_k = int(full_ids[prompt_len + k])
                    # KV cache currently holds prompt_len + (k-1) + 1 tokens
                    # (last append + sampled). Rewind sampled tail to keep the
                    # cache in lock-step with the actual sequence we want.
                    generator.rewind_to(prompt_len + k - 1)
                    generator.append_tokens([prev_target])
                    generator.generate_next_token()
                    logits_step = np.asarray(generator.get_output("logits"))
                    if logits_step.ndim == 3:
                        logits_step = logits_step[0]
                    logits_t = torch.from_numpy(logits_step.astype(np.float32))
                    step_logits = logits_t[-1:, :]
                    step_label = torch.tensor([target_k], dtype=torch.int64)
                    step_nll = loss_fct(step_logits, step_label).item()
                    if not np.isfinite(step_nll):
                        if verbose:
                            print(f"  [diag] non-finite NLL at decode step "
                                  f"{k} (target_id={target_k}); aborting sample")
                        return None
                    sum_nll += step_nll
                    _record_step(step_logits[-1], target_k, step_nll)
        finally:
            del generator

        rec = {
            "target_ids": tgt_ids_out,
            "target_nll": per_tok_out,
            "top1_ids": top1_out,
        }
        if topk and topk > 0:
            rec["target_logit"] = target_logit_out
            rec["lse"] = lse_out
            rec["topk_ids"] = topk_ids_out
            rec["topk_logits"] = topk_logits_out
        return sum_nll, target_len, rec
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(args):
    print(f"Calculating VLM Perplexity on dataset={args.dataset} "
          f"split={args.split} limit={args.limit} mode={args.mode} ...")

    if args.verbose:
        print("Loading dataset ...")
    if args.dataset == "lmms-lab/flickr30k" and args.split == "test":
        samples = load_flickr30k_test(args.limit)
    else:
        # Generic HF path for future datasets — kept simple on purpose.
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split=args.split)
        samples = []
        for ex in ds:
            if len(samples) >= args.limit:
                break
            img = ex.get("image", None)
            cap = ex.get("caption", None)
            if isinstance(cap, list) and cap:
                cap = cap[0]
            if not isinstance(img, Image.Image) or not isinstance(cap, str):
                continue
            samples.append((img.convert("RGB"), cap.strip()))
        if not samples:
            raise RuntimeError(f"No usable samples from {args.dataset}/{args.split}")
    if args.verbose:
        print(f"Loaded {len(samples)} samples")

    if args.verbose:
        print("Loading model (this may JIT-compile EPs on first call) ...")
    model = og.Model(args.model)
    tokenizer = og.Tokenizer(model)
    processor = model.create_multimodal_processor()
    template_str = load_chat_template(args.model)
    if args.verbose:
        print(f"Model loaded; chat_template "
              f"{'found' if template_str else 'missing'}")

    total_nll = 0.0
    total_tokens = 0
    skipped = 0
    per_sample_ppl: List[float] = []

    # Optional per-token dump for the EP-vs-CPU comparison driver
    # (compare_vlm_ep_cpu.py). One JSON object per evaluated sample carrying
    # the target token ids, per-token NLL, and per-token top-1 prediction.
    dump_path = getattr(args, "dump_jsonl", None)
    dump_fh = open(dump_path, "w", encoding="utf-8") if dump_path else None

    # ``mode`` chooses which image (if any) is paired with each caption.
    # The CAPTION is always the canonical caption[0] of sample i — only the
    # IMAGE is rotated. This isolates the visual-conditioning effect.
    n = len(samples)
    if args.mode == "match":
        pair_img_for = lambda i: samples[i][0]
    elif args.mode == "shuffle":
        # Pair caption_i with image_(i+1) mod N. A constant non-zero offset
        # is reproducible and guarantees img ≠ caption's source for every
        # sample (so long as N ≥ 2).
        pair_img_for = lambda i: samples[(i + 1) % n][0]
    elif args.mode == "none":
        pair_img_for = lambda i: None  # text-only path inside compute_sample_nll
    else:
        raise ValueError(f"unknown mode {args.mode!r}")
    if args.verbose:
        print(f"Visual-grounding mode: {args.mode}")

    # Optional sample-index filter: only evaluate the listed dataset indices
    # (keeps the original index numbering so dumps align with a full run).
    selected = None
    sidx = getattr(args, "sample_indices", None)
    if sidx:
        selected = [int(x) for x in str(sidx).split(",") if x.strip() != ""]
    indices = selected if selected is not None else list(range(len(samples)))
    topk = int(getattr(args, "dump_topk", 0) or 0)

    for i in indices:
        if i < 0 or i >= len(samples):
            if args.verbose:
                print(f"  [skip] sample index {i} out of range "
                      f"(have {len(samples)} samples)")
            continue
        caption = samples[i][1]
        if args.verbose:
            print(f"[{i + 1}/{len(samples)}] caption={caption[:60]!r}")
        result = compute_sample_nll(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            template_str=template_str,
            img=pair_img_for(i),
            caption=caption,
            instruction=args.instruction,
            image_size=args.image_size,
            max_length=args.max_length,
            verbose=args.verbose,
            topk=topk,
        )
        if result is None:
            skipped += 1
            continue
        sum_nll, n_target, rec = result
        # Drop samples whose NLL came out non-finite (NaN/Inf). Causes seen so
        # far are: an Inf logit at some target position (rare numeric edge in
        # the prefill) or token-shift mis-alignment past EOS. A single NaN
        # would otherwise corrupt total_nll, making the final PPL == NaN.
        if not np.isfinite(sum_nll):
            if args.verbose:
                print(f"  [skip] sum_nll non-finite ({sum_nll}); "
                      f"n_target={n_target}")
            skipped += 1
            continue
        total_nll += sum_nll
        total_tokens += n_target
        sample_ppl = float(np.exp(sum_nll / max(n_target, 1)))
        per_sample_ppl.append(sample_ppl)
        if dump_fh is not None:
            row = {"index": i, "target_len": int(n_target),
                   "sum_nll": float(sum_nll)}
            row.update(rec)
            dump_fh.write(json.dumps(row) + "\n")
            dump_fh.flush()
        if args.verbose:
            print(f"  sum_nll={sum_nll:.4f} n_target={n_target} "
                  f"sample_ppl={sample_ppl:.4f}")

    if dump_fh is not None:
        dump_fh.close()

    if total_tokens == 0:
        print("ERROR: every sample was skipped; cannot compute PPL", file=sys.stderr)
        sys.exit(2)

    avg_nll = total_nll / total_tokens
    ppl = float(np.exp(avg_nll))

    # The summary block doubles as the parser target for the wrapper.
    print("=" * 60)
    n_attempted = len(indices)
    print(f"Samples evaluated : {n_attempted - skipped}/{n_attempted}")
    print(f"Skipped           : {skipped}")
    print(f"Total target tokens: {total_tokens}")
    print(f"Average NLL       : {avg_nll:.6f}")
    if per_sample_ppl:
        per = np.array(per_sample_ppl, dtype=np.float64)
        print(f"Per-sample PPL    : "
              f"min={per.min():.4f} mean={per.mean():.4f} "
              f"median={np.median(per):.4f} max={per.max():.4f}")
    # IMPORTANT: `Perplexity: <value>` is the line tests/ppl.py /
    # tests/ppl_vlm.py regex-parses. Keep this format stable.
    print(f"Perplexity: {ppl:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
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
    parser.add_argument("--max_length", type=int, default=384,
                        help="OGA search.max_length used for KV buffer sizing. "
                             "Default 384; raise for longer captions.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("--dump-jsonl", dest="dump_jsonl", type=str, default=None,
                        help="Write per-sample per-token data (target_ids, "
                             "target_nll, top1_ids) as JSONL for the EP-vs-CPU "
                             "comparison driver (compare_vlm_ep_cpu.py).")
    parser.add_argument("--dump-topk", dest="dump_topk", type=int, default=0,
                        help="If >0, also dump per-target-position top-K "
                             "(token id + logit), target_logit and logsumexp "
                             "for logit-level EP-vs-CPU divergence analysis.")
    parser.add_argument("--sample-indices", dest="sample_indices", type=str,
                        default=None,
                        help="Comma-separated dataset indices to evaluate "
                             "(e.g. '40,16,6'). Others are skipped; original "
                             "index numbering is preserved in the dump.")
    parser.add_argument("--mode", type=str, default="match",
                        choices=["match", "shuffle", "none"],
                        help="Visual-grounding mode. 'match': caption_i paired "
                             "with its own image_i (canonical PPL). 'shuffle': "
                             "caption_i paired with image_(i+1)%N (visual-"
                             "neglect probe). 'none': caption_i with no image "
                             "(language-prior baseline). Run all three to "
                             "compute PPL_shuf/PPL_match and PPL_none/PPL_match "
                             "ratios — both should be > 1 for a healthy VLM.")
    # accept (and ignore) -d/--device for symmetry with perplexity.py callers
    parser.add_argument("--device", type=str, required=False, default="cpu",
                        choices=["cpu", "aie", "gpu"], help="(ignored)")
    args = parser.parse_args()

    # argparse SUPPRESS leaves missing optional args off the namespace; rebind
    # the defaults we always want present.
    for k, v in [
        ("dataset", "lmms-lab/flickr30k"),
        ("split", "test"),
        ("limit", 50),
        ("instruction", "Describe this image briefly."),
        ("image_size", 896),
        ("max_length", 384),
        ("verbose", False),
        ("mode", "match"),
        ("dump_jsonl", None),
        ("dump_topk", 0),
        ("sample_indices", None),
    ]:
        if not hasattr(args, k):
            setattr(args, k, v)
    main(args)
