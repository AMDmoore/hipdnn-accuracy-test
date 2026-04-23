import os
import sys
if os.name == 'nt':
    cache_path = "r:/aiebuilds/tools/py312_cache/win64"
else:
    cache_path = "/proj/aiebuilds/tools/py312_cache/lnx64"
    
sys.path.append(cache_path)
import single_mmlu_eval_vitis_ref as mmlu_eval
import torch
from colorama import Fore
import argparse

# For OGA

print(Fore.GREEN+"Computing MMLU..."+Fore.RESET)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end token generation loop example for gen-ai")
    parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-c', '--class_', type=str, required=False, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-s', '--start', type=str, required=False,default=0, help='Onnx model folder path (must contain config.json and model.onnx)')
    parser.add_argument('-e', '--end', type=str, required=False,default=0, help='Onnx model folder path (must contain config.json and model.onnx)')
    
    args = parser.parse_args()
    mmlu_eval.mmlu(args.model,args.class_,args.start,args.end, framework="oga")