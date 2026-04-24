import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmlu_eval_vitis_ref as mmlu_eval
import torch
from colorama import Fore
import argparse

# For OGA

print(Fore.GREEN+"Computing MMLU..."+Fore.RESET)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end token generation loop example for gen-ai")
    parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-l', '--max-input-len', type=int, default=None, help='Max input length (seqlen / fixed_prompt_length)')
    parser.add_argument('-c', '--context-length', type=int, default=None, help='Context length (max_length for OGA)')
    args = parser.parse_args()
    mmlu_eval.mmlu(args.model, framework="oga",
                   max_input_len_override=args.max_input_len,
                   context_length_override=args.context_length)