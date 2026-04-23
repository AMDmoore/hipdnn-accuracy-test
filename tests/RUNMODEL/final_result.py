import os
import sys

import pandas as pd
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified script for ChatGLM and Llama ONNX model inference.")
    parser.add_argument("-m", "--model_name", type=str, required=True,
                        help="Path to ONNX model folder (must contain genai_config.json and model.onnx)") 
    args = parser.parse_args()
                        
    model = args.model_name

final_dict =[]
result_model={"model":model}
with open(f"output.json","r") as f:
	data = json.load(f)
result = data["generations"]
result_df = pd.DataFrame(result)
result_df=result_df[["prompt","response"]]
result_dict = result_df.set_index("prompt")['response'].to_dict()
result_model.update(result_dict)
final_dict.append(result_model)
final_result = pd.DataFrame(final_dict)
#print(final_result)
final_result.to_csv("output.csv",index=False)
