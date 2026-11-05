import onnxruntime_genai as og
import argparse
import os
import time
import random
import numpy as np
import torch
import json
import glob
import hashlib

from start_prefix import MODES as BOS_MODES, resolve_start_prefix

def get_wikitext2(tokenizer, dataset="non-raw"):
    """Encode the wikitext2 test split with the OGA tokenizer.

    The stream is encoded without special tokens; the per-window start prefix
    is added by the PPL loops (see start_prefix.py)."""
    from datasets import load_dataset

    if dataset == "non-raw":
        testdata = load_dataset("wikitext", "wikitext-2-v1", split="test")
    elif dataset == "raw":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        raise ValueError(
            "You are using an unsupported dataset, only support wikitext2-raw-v1 and wikitext2-v1."
            "Using wikitext2-raw-v1 with --dataset=raw and wikitext2-v1 with --dataset=non-raw."
        )
    tokenizer.update_options(add_special_tokens="false")
    return np.asarray(tokenizer.encode("\n\n".join(testdata["text"])), dtype=np.int32)

def with_prefix(prefix_ids, tokens):
    """Prepend the start prefix to a window of corpus tokens."""
    if not prefix_ids:
        return tokens
    return np.concatenate((np.asarray(prefix_ids, dtype=np.int32), tokens))

def make_window(prefix_ids, tokens, seqlen, verbose=False, first=False):
    """Build one model input: the start prefix followed by ``seqlen`` corpus tokens."""
    window = with_prefix(prefix_ids, tokens)
    p = len(prefix_ids)
    assert len(window) == seqlen + p, (
        f"window has {len(window)} tokens, expected {seqlen} + {p} prefix")
    assert list(window[:p]) == list(prefix_ids), "window does not start with the prefix"
    if verbose and first:
        print(f"[window] first window: {len(window)} tokens, head ids {[int(t) for t in window[:p + 3]]}")
    return window

def detect_pruned_logits(model, params, test_enc, seqlen):
    """Probe once whether the decoder emits full-sequence logits [B,T,V] or
    only last-position logits [B,1,V].

    The prune-logits transform inserts Gather(idx=-1)+Unsqueeze before lm_head,
    so a pruned graph returns a single position regardless of input length. A
    tiny probe is enough to tell the two apart: feed N>=2 tokens and check the
    logits sequence dimension (== N for full logits, == 1 for pruned)."""
    probe_len = min(len(test_enc), max(2, min(int(seqlen), 16)))
    generator = og.Generator(model, params)
    generator.append_tokens(test_enc[:probe_len])
    logits = np.asarray(generator.get_output("logits"))
    del generator
    if logits.ndim == 3:
        seq_dim = logits.shape[1]
    elif logits.ndim == 2:
        seq_dim = logits.shape[0]
    else:
        seq_dim = 1
    return seq_dim < probe_len

def run_full_logits_ppl(model, params, test_enc, seqlen, prefix_ids, nsamples_frac, verbose):
    """Non-pruned path: full-sequence logits per chunk.

    The corpus is partitioned into disjoint chunks of ``seqlen`` corpus tokens,
    so chunk boundaries do not depend on the start prefix. Each model input is
    ``prefix + chunk`` and is prefilled once.

    Scoring is defined from ``seqlen``, not from the prefix length: the last
    ``seqlen`` logits rows line up with the chunk, and rows [-seqlen:-1]
    predict chunk[1:]. The first chunk token is never scored (it has no real
    context, and the prefix row that would predict it is dropped), so every
    window scores exactly seqlen-1 tokens with or without a prefix."""
    n_scored = seqlen - 1
    if n_scored < 1:
        raise ValueError(f"seqlen={seqlen} leaves no scored tokens")
    nsamples = int(nsamples_frac * (len(test_enc) // seqlen))
    if nsamples < 1:
        raise ValueError(
            f"nsamples resolved to {nsamples}; increase -n or shorten -l "
            f"(len(test_enc)={len(test_enc)}, seqlen={seqlen})")
    if verbose:
        print(f"[full-logits] nsamples (chunks): {nsamples}")
        print(f"nbr of tokens in testenc: {len(test_enc)}")
        print(f"seqlen: {seqlen} (+ prefix {len(prefix_ids)} per window)")
        print("Running generation loop ...")
    nlls = []
    with torch.no_grad():
        for i in range(nsamples):
            window = make_window(prefix_ids, test_enc[(i * seqlen) : ((i + 1) * seqlen)],
                                 seqlen, verbose, first=(i == 0))
            generator = og.Generator(model, params)
            generator.append_tokens(window)
            logits = torch.tensor(generator.get_output("logits")[0], dtype=torch.float32)
            # With chunked prefill, get_output("logits") holds only the last
            # chunk's rows (earlier chunks are overwritten), so the row count
            # would silently fall short and mis-slice the scored rows.
            if logits.shape[0] < seqlen:
                raise RuntimeError(
                    f"full-logits PPL needs {seqlen} logits rows per window but the "
                    f"model returned {logits.shape[0]}; prefill chunking "
                    f"(search.chunk_size) keeps only the last chunk's logits")
            # Rows [-seqlen:-1] sit on chunk[0..seqlen-2] and predict chunk[1:],
            # the last seqlen-1 tokens of the window.
            shift_logits = logits[-seqlen:-1]
            shift_labels = torch.tensor(window[-n_scored:], dtype=torch.long)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)
            neg_log_likelihood = loss.float() * n_scored
            nlls.append(neg_log_likelihood)
            del generator
            print(f"Iteration {i+1} / {nsamples} done", end ='\r')
        return torch.exp(torch.stack(nlls).sum() / (nsamples * n_scored))

def run_pruned_ppl(model, params, test_enc, seqlen, prefix_ids, stride, nsamples_frac, verbose):
    """Pruned path: last-position-only logits via a strided sliding window.

    A pruned graph emits logits only for the final input position, so each
    prefill yields exactly one usable prediction. We slide a window over the
    corpus by ``stride`` and, per window, score the single corpus token that
    immediately follows it. The window slicing and labels are defined on the
    corpus and do not depend on the start prefix; the model input is
    ``prefix + seqlen corpus tokens``, and the last row always belongs to the
    last corpus token of the window:

        window i = prefix + test_enc[i*stride : i*stride + seqlen]  (a real prefill)
        logits   = last-position logits = P(? | window i)
        label    = test_enc[i*stride + seqlen]

    Every scored token therefore has a full ``seqlen`` corpus context. A fresh
    Generator per window guarantees a real prefill (no KV reuse / decode path).
    The number of windows is derived from ``nsamples`` (a fraction of the
    corpus), mirroring the non-pruned nsamples semantics."""
    if len(test_enc) <= seqlen:
        raise ValueError(
            f"len(test_enc)={len(test_enc)} must exceed seqlen={seqlen} for the "
            f"pruned sliding-window PPL; shorten -l or use a longer dataset")

    total_sample = int(nsamples_frac * len(test_enc))
    real_sample = (total_sample + seqlen) if total_sample <= seqlen else total_sample
    real_sample = min(real_sample, len(test_enc))
    num_windows = (real_sample - seqlen) // stride + 1
    # Cap so the last window's label position (seqlen + (n-1)*stride) stays in range.
    max_windows = (len(test_enc) - seqlen - 1) // stride + 1
    num_windows = max(1, min(num_windows, max_windows))

    if verbose:
        print(f"[pruned] sliding window: stride={stride}, seqlen={seqlen} (+ prefix {len(prefix_ids)} per window)")
        print(f"nbr of tokens in testenc: {len(test_enc)}")
        print(f"[pruned] nsamples (windows) = {num_windows} window prefills")
        print("Running generation loop ...")

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        tensor_arrays = []
        for i in range(num_windows):
            input_tokens = make_window(prefix_ids, test_enc[i * stride : (i * stride + seqlen)],
                                       seqlen, verbose, first=(i == 0))
            generator = og.Generator(model, params)
            generator.append_tokens(input_tokens)
            # [0] -> batch 0; [-1] -> last position (the only one for pruned).
            logits = generator.get_output("logits")[0][-1].reshape((1, -1))
            tensor_arrays.append(torch.tensor(logits, dtype=torch.float32))
            del generator
            print(f"Iteration {i+1} / {num_windows} done", end='\r')

        logits_tensor = torch.cat(tensor_arrays)
        label_tensor = torch.tensor(
            test_enc[seqlen : (seqlen + num_windows * stride) : stride],
            dtype=torch.long)
        if verbose:
            print(f"logits_tensor shape={tuple(logits_tensor.shape)} "
                  f"label_tensor shape={tuple(label_tensor.shape)}")
        loss = cross_entropy_loss(logits_tensor, label_tensor)
        return torch.exp(loss / num_windows)

def main(args):
    # Plugin EP registration is handled by OGA itself from the <EP>_EP_PATH env
    # var (exported by setup_package_env and inherited by this subprocess); no
    # explicit register_execution_provider_library call is needed here.
    print(f"Calculating Perplexity on wikitext2 test set ...")
    if args.verbose: print("Loading model...")
    model = og.Model(f'{args.model}')
    if args.verbose: print("Model loaded")

    # The OGA tokenizer is used for every dataset so the token stream matches
    # what the runtime feeds the model.
    tokenizer = og.Tokenizer(model)
    if args.verbose: print("Tokenizer created")

    test_enc = get_wikitext2(tokenizer, dataset=args.dataset)
    if args.verbose: print("Dataset acquired")

    # The corpus carries no special tokens. Every window is the start prefix
    # (the ids the tokenizer adds at the start of a sequence when special
    # tokens are enabled; empty for tokenizers that add none) followed by
    # exactly -l corpus tokens, so slicing and labels are prefix-independent.
    start_prefix = resolve_start_prefix(tokenizer, args.bos)
    prefix_ids = list(start_prefix.ids)
    print(f"[start-prefix] mode={args.bos} source={start_prefix.source} ids={prefix_ids}")
    print(f"[corpus] tokens={len(test_enc)} "
          f"sha256={hashlib.sha256(test_enc.tobytes()).hexdigest()[:16]}")

    # set generator parameters
    params = og.GeneratorParams(model)
    search_options = {name:getattr(args, name) for name in ['do_sample', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}

    if not hasattr(args, 'max_length'):
        raise ValueError("-l (seqlen / fixed_prompt_length) is required for PPL")
    seqlen = args.max_length
    if hasattr(args, 'context_length') and args.context_length is not None:
        # KV capacity must hold the whole window: prefix + seqlen corpus tokens.
        search_options['max_length'] = max(args.context_length, seqlen + len(prefix_ids))
    params.set_search_options(**search_options)
    if hasattr(params, 'try_graph_capture_with_max_batch_size'):
        params.try_graph_capture_with_max_batch_size(1)
    if args.verbose: print("GeneratorParams created")

    assert args.nsamples<=1.0, "nsamples must be less than 1!"

    # A prune-logits model emits only last-position logits, which breaks the
    # full-sequence chunk path (the shifted logits rows would be missing). Probe
    # once and pick the matching PPL strategy:
    #   - full logits  -> disjoint chunks, every position scored from one prefill.
    #   - pruned       -> strided sliding-window, one last-token score per window
    #                     (still a real prefill per window).
    pruned = detect_pruned_logits(model, params, test_enc, seqlen)
    if args.verbose:
        print(f"[detect] logits mode: {'PRUNED (last-position only)' if pruned else 'FULL sequence'}")

    if pruned:
        stride = int(getattr(args, 'stride', 256))
        ppl = run_pruned_ppl(model, params, test_enc, seqlen, prefix_ids, stride,
                             args.nsamples, args.verbose)
    else:
        # Scoring needs every position's logits from one prefill. A model config
        # may enable chunked prefill (search.chunk_size), which keeps only the
        # last chunk's logits; disable it for this run. Chunking only splits the
        # prefill, so the computed logits are unchanged. The pruned path is not
        # affected: its single output row is the last position either way.
        params.set_search_options(chunk_size=0)
        ppl = run_full_logits_ppl(model, params, test_enc, seqlen, prefix_ids,
                                  args.nsamples, args.verbose)
    print("Perplexity:", ppl.item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end token generation loop example for gen-ai")
    parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, help='Window length in corpus tokens; the start prefix (if any) is added on top of it')
    parser.add_argument('-ds', '--do_random_sampling', action='store_true', help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output and timing information. Defaults to false')
    parser.add_argument('-n', '--nsamples', type=float, default = 1.0, help='Number of samples of wikitext2 to use in computing the perplexity')
    parser.add_argument('-d', "--device", required=False, default="cpu", choices=["cpu", "aie"], help="Target device (CPU or Ryzen-AI)")
    parser.add_argument('-s', "--dataset", required=False, default="raw", choices=["raw", "non-raw"], help="Wikitext2 dataset version (raw or non-raw). Defaults to 'raw'")
    parser.add_argument('-c', '--context-length', type=int, default=None, help='Context length (max_length for OGA); raised to fit prefix + -l if smaller. If not set, reads from genai_config.json')
    parser.add_argument('--stride', type=int, default=256, help='Sliding-window stride for the pruned (last-position-only) PPL path. Ignored for full-logits models. Default 256')
    parser.add_argument('--bos', choices=BOS_MODES, default='auto', help="Sequence-start prefix for every window. 'auto' prepends whatever the OGA tokenizer adds at the start of a sequence when special tokens are enabled (nothing for tokenizers that add none); 'never' disables it. Default auto")
    args = parser.parse_args()
    main(args)
