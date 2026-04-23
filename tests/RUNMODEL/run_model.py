# Copyright Â© 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import argparse
import json
import time
import numpy as np
import onnxruntime_genai as og
from tokenizer_factory import get_tokenizer
sys.stdout.reconfigure(encoding='utf-8')

# Default prompts if file is missing or incorrect
DEFAULT_PROMPTS = [
    "Write a one-line definition of gravity",
    "Do not explain. Just give the number: 15 * 7",
    "A shop gives 10% discount on 2000 Rupees. What is final price?",
    "Tell me about the invention of flying cars in 1800s",
]

def load_prompts(prompt_input):
    """Loads prompts from a .txt file, a direct string, or falls back to default prompts."""
    if prompt_input:
        if os.path.exists(prompt_input):
            with open(prompt_input, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f.readlines() if line.strip()]
            if prompts:
                return prompts
        else:
            # Treat as a direct string input
            return [prompt_input]
    print("Warning: Invalid or missing prompt input. Using default prompts.")
    return DEFAULT_PROMPTS

def sanitize_string(input_string):
    return input_string.encode("charmap", "ignore").decode("charmap")

def load_model_and_tokenizer(model_path, verbose=False):
    """Loads the ONNX model and tokenizer, determining the model type from the config."""
    config_path = os.path.join(model_path, 'genai_config.json')

    # Read the model type from the configuration file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        model_type = config['model']['type']

    if verbose:
        print(f"Loading {model_type} model from {model_path}...")

    model = og.Model(model_path)
    tokenizer = get_tokenizer(model_path, model_type, model)
    return model, tokenizer, model_type

def generate_text(model, tokenizer, prompts, model_type, args):
    """Handles text generation for both models."""
    params = og.GeneratorParams(model)
    search_options = {
        "do_sample": args.do_random_sampling,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
    }
    params.set_search_options(**{k: v for k, v in search_options.items() if v is not None})
    if hasattr(params, "try_graph_capture_with_max_batch_size"):
        params.try_graph_capture_with_max_batch_size(len(prompts))
    #params.try_graph_capture_with_max_batch_size(len(prompts))

    total_tokens = 0
    results = []

    print("\nGenerating responses...\n")

    for i, prompt in enumerate(prompts):
        start_time = time.time()
        generator=og.Generator(model, params)
        generator.append_tokens(tokenizer.encode(prompt))
        while not generator.is_done():
            generator.generate_next_token()
        output_tokens = generator.get_sequence(0)
        elapsed_time = time.time() - start_time

        output_text = tokenizer.decode(output_tokens)
        
        print(f"Prompt #{i+1}: {prompt}")
        print("Output:", sanitize_string(output_text))
        #print("Output:", output_text.encode('ascii', errors='replace').decode('ascii'))
        print(f"Time taken: {elapsed_time:.2f}s | Tokens generated: {len(output_tokens)}\n")

        results.append({
            "prompt": prompt,
            "response": output_text,
            "tokens": len(output_tokens)
        })

        total_tokens += len(output_tokens)

    print(f"Total tokens generated: {total_tokens}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump({"generations": results}, f, indent=2)
        print(f"Results saved to: {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Unified script for ChatGLM and Llama ONNX model inference.")

    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="Path to ONNX model folder (must contain genai_config.json and model.onnx)")
    parser.add_argument("-pr", "--prompt_file", type=str,
                        help="Path to .txt file containing prompts (one per line) or direct input string")
    parser.add_argument("-o", "--output_file", type=str,
                        default='output.json',
                        help="Path to output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output.")

    # Generation parameters
    parser.add_argument("-l", "--max_length", type=int,
                        help="Max tokens to generate including the prompt")
    parser.add_argument("-i", "--min_length", type=int,
                        help="Min tokens to generate including the prompt")
    parser.add_argument("-ds", "--do_random_sampling", default=False,
                        action="store_true",
                        help="Enable random sampling instead of greedy search")
    parser.add_argument("-p", "--top_p", type=float,
                        help="Top p probability for nucleus sampling")
    parser.add_argument("-k", "--top_k", type=int,
                        help="Top k tokens to sample from")
    parser.add_argument("-temp", "--temperature", type=float,
                        help="Sampling temperature")
    parser.add_argument("-r", "--repetition_penalty", type=float,
                        help="Penalty for repeated words")

    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts(args.prompt_file)

    # Load model and tokenizer, determine model type from the config
    model, tokenizer, model_type = load_model_and_tokenizer(args.model_path, args.verbose)

    # Run generation
    generate_text(model, tokenizer, prompts, model_type, args)

if __name__ == "__main__":
    main()