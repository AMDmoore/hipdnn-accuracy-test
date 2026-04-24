import onnxruntime_genai as og
import argparse
import os
import time
import random
import numpy as np
import torch
import json
import glob
from transformers import AutoTokenizer, LlamaTokenizer

def get_wikitext2(tokenizer, dataset="non-raw"):
    """gptq"""
    from datasets import load_dataset

    if dataset == "non-raw":
        traindata = load_dataset("wikitext", "wikitext-2-v1", split="train")
        testdata = load_dataset("wikitext", "wikitext-2-v1", split="test")
    elif dataset == "raw":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        raise ValueError(
            "You are using an unsupported dataset, only support wikitext2-raw-v1 and wikitext2-v1."
            "Using wikitext2-raw-v1 with --dataset=raw and wikitext2-v1 with --dataset=non-raw."
        )
    if dataset=="non-raw":
        trainenc = torch.tensor(tokenizer.encode("\n\n".join(traindata["text"])))
        testenc = torch.tensor(tokenizer.encode("\n\n".join(testdata["text"])))
    else:
        train_enc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        test_enc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        testenc = np.squeeze(test_enc.input_ids)
    dataloader = []
    # for _ in range(nsamples):
    #     i = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = testenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     dataloader.append((inp, tar))
    return dataloader, testenc

def main(args):
    # Compute perplexity using the sum of decomposed log-likelihoods of disjoint chunks of the dataset
    # Plugin EP registration (MorphiZenEP, etc.) is handled centrally by
    # tests/_ep_bootstrap.py, which the orchestrator injects in front of
    # this script's invocation.
    print(f"Calculating Perplexity on wikitext2 test set ...")
    if args.verbose: print("Loading model...")
    model = og.Model(f'{args.model}')
    if args.verbose: print("Model loaded")

    # create the tokenizer (oga tokenizer cannot encode raw dataset)
    if args.dataset=="non-raw":
        tokenizer = og.Tokenizer(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f'{args.model}', token=False, use_fast=True, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    if args.verbose: print("Tokenizer created")

    #load the dataset
    dataloader, test_enc = get_wikitext2(tokenizer, dataset=args.dataset)
    if args.verbose: print("Dataset acquired")

    # set generator parameters
    params = og.GeneratorParams(model)
    search_options = {name:getattr(args, name) for name in ['do_sample', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}

    if hasattr(args, 'max_length'):
        seqlen = args.max_length
    else:
        seqlen = args.context_length
    search_options['max_length'] = args.context_length
    params.set_search_options(**search_options)
    if hasattr(params, 'try_graph_capture_with_max_batch_size'):
        params.try_graph_capture_with_max_batch_size(1)
    if args.verbose: print("GeneratorParams created")

    # the dataset is partitioned into nsamples sequences of length seqlen
    assert args.nsamples<=1.0, "nsamples must be less than 1!"
    nsamples = int(args.nsamples * (len(test_enc) // seqlen))

    if args.verbose:
        print(f"nsamples: {nsamples}")
        print(f"nbr of tokens in testenc: {len(test_enc)}")
        print(f"seqlen: {seqlen}")

    if args.verbose: print("Running generation loop ...")
    loss = torch.nn.CrossEntropyLoss()
    nlls = []
    # iterate over the nsamples sequences
    with torch.no_grad():
        for i in range(nsamples):
            input_tokens = test_enc[(i * seqlen) : ((i + 1) * seqlen)]
            generator = og.Generator(model, params)
            generator.append_tokens(input_tokens)
            logits = torch.tensor(generator.get_output("logits")[0], dtype=torch.float32)
            # standard shift: logits[:-1] predicts tokens[1:]
            shift_logits = logits[:-1]
            shift_labels = torch.tensor(test_enc[(i * seqlen) + 1 : ((i + 1) * seqlen)], dtype=torch.long)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)
            neg_log_likelihood = loss.float() * (seqlen - 1)
            nlls.append(neg_log_likelihood)
            del generator
            print(f"Iteration {i+1} / {nsamples} done", end ='\r')
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
        print("Perplexity:", ppl.item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end token generation loop example for gen-ai")
    parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-ds', '--do_random_sampling', action='store_true', help='Do random sampling. When false, greedy or beam search are used to generate the output. Defaults to false')
    parser.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-r', '--repetition_penalty', type=float, help='Repetition penalty to sample with')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output and timing information. Defaults to false')
    parser.add_argument('-n', '--nsamples', type=float, default = 1.0, help='Number of samples of wikitext2 to use in computing the perplexity')
    parser.add_argument('-d', "--device", required=False, default="cpu", choices=["cpu", "aie"], help="Target device (CPU or Ryzen-AI)")
    parser.add_argument('-s', "--dataset", required=False, default="raw", choices=["raw", "non-raw"], help="Wikitext2 dataset version (raw or non-raw). Defaults to 'raw'")
    parser.add_argument('-c', '--context-length', type=int, default=None, help='Context length (max_length for OGA). If not set, reads from genai_config.json')
    args = parser.parse_args()
    if args.context_length is None:
        with open(os.path.join(args.model, 'genai_config.json')) as f:
            genai_cfg = json.load(f)
        args.context_length = genai_cfg["model"]["context_length"]
    main(args)
