"""Generate OGA responses for TinyGSM8K math problems.

Phase 1 of the TINYGSM8K accuracy test: reads math questions from an input
JSON file, generates responses via onnxruntime-genai, and writes them to an
output text file separated by an end-of-response marker.

Adapted from llm_accuracy_scripts_win/TINYGSM8K/oga_responses_generation.py.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

import onnxruntime_genai as og

PSU_PROMPT = (
    "Please solve following problem and explain it to me. "
    "Then give me final answer at the end with a single number "
    "preceded by string '#### '. "
)


def set_seeds(random_seed=0, numpy_seed=1234, torch_seed=1234):
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)


def generate(model_dir, inputs, output_file, context_length=None,
             max_new_tokens=512, case="psu_prompt_eos_stop", eor="<EOR>",
             verbose=False):
    with open(os.path.join(model_dir, "genai_config.json"), "r") as f:
        config = json.load(f)
    eos_token_id = config["model"]["eos_token_id"]
    if context_length is None:
        context_length = config["model"]["context_length"]

    set_seeds()

    if verbose:
        print(f"Loading model from {model_dir}...")
    model = og.Model(model_dir)
    tokenizer = og.Tokenizer(model)

    outputs = []
    correct_format = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for i in tqdm(range(len(inputs)), desc="Generating"):
            if case == "default":
                prompt = inputs[i]
            else:
                prompt = PSU_PROMPT + inputs[i]

            input_tokens = tokenizer.encode(prompt)

            params = og.GeneratorParams(model)
            params.set_search_options(max_length=context_length)

            generator = og.Generator(model, params)
            generator.append_tokens(input_tokens)

            response = ""
            tokenizer_stream = tokenizer.create_stream()
            num_tokens = 0
            while not generator.is_done():
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                if case == "psu_prompt_eos_stop" and new_token == eos_token_id:
                    break
                response += tokenizer_stream.decode(new_token)
                num_tokens += 1
                if num_tokens >= max_new_tokens:
                    break

            del generator

            if "####" in response:
                correct_format += 1

            f.write(response + f"\n{eor}\n")
            f.flush()
            outputs.append(response)

            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(inputs)}] generated {num_tokens} tokens")

    print(f"Responses saved to: {output_file}")
    print(f"Correct format (contains '####'): {correct_format}/{len(inputs)}")
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate OGA responses for TinyGSM8K")
    parser.add_argument("-m", "--model-dir", type=str, required=True)
    parser.add_argument("-i", "--inputs", type=str, required=True,
                        help="Path to tinyGSM8k_inputs JSON file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output text file for generated responses")
    parser.add_argument("-c", "--context-length", type=int, default=None,
                        help="Max generation length (defaults to genai_config context_length)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max new tokens per question (default: 512)")
    parser.add_argument("--case", type=str, default="psu_prompt_eos_stop",
                        choices=["default", "psu_prompt", "psu_prompt_eos_stop"])
    parser.add_argument("--eor", type=str, default="<EOR>",
                        help="End-of-response separator")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.inputs, "r") as f:
        inputs = json.load(f)
    print(f"Loaded {len(inputs)} inputs from {args.inputs}")

    generate(
        model_dir=args.model_dir,
        inputs=inputs,
        output_file=args.output,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        case=args.case,
        eor=args.eor,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
