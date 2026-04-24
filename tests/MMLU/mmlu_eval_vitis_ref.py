import os
import sys
import re
import gzip
import json
import logging
from typing import List
import random
import time
from collections import defaultdict
from typing import Dict, Iterable
import pandas as pd
from thefuzz import process
from tqdm import tqdm
import transformers
from transformers.trainer_utils import set_seed
import onnxruntime_genai as og
from transformers import AutoTokenizer
import torch
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    BloomForCausalLM,
    MistralForCausalLM,
    OPTForCausalLM,
    PhiForCausalLM,
  #  SinkCache,
    TextStreamer,
    AutoTokenizer,
)

def mmlu(model, framework="pytorch", max_new_tokens=50, nsamples=30):
    # OGA fixed-shape models hardcode their input window and total token
    # budget in the model's own genai_config.json (which lives alongside
    # model.onnx and *is* the model's metadata — model.onnx itself is
    # symbolic-shape). Read both upfront so get_logits() can pick the
    # right truncation length and the right max_length.
    max_input_len = 4096            # pytorch fallback
    max_length = None               # OGA-only; required if framework=='oga'
    if framework == "oga":
        with open(os.path.join(model, "genai_config.json"),
                  "r", encoding="utf-8") as _f:
            _mcfg = json.load(_f)["model"]
        _dec = _mcfg.get("decoder", {})
        max_input_len = int(
            _dec.get("fixed_prompt_length")
            or _dec.get("sliding_window", {}).get("window_size")
            or _mcfg["context_length"]
        )
        max_length = int(_mcfg["context_length"])
        print(f"  [MMLU] max_input_len={max_input_len}, "
              f"max_length={max_length} (from genai_config.json)")


    # set dataset path
    # if not (os.path.exists("./mmlu_data")):
    #     print("Retrieving data...")
    #     url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    #     urllib.request.urlretrieve(url, "data.tar")
    #     file = tarfile.open("data.tar")
    #     file.extractall("./mmlu_data")
    #     file.close()
    #     print("Data retrieved")
    
    eval_data_path = "./mmlu_data"
    if framework == "oga":
        model_name = os.path.abspath(model).split("\\")[-1]
        # with open(f'{model}\genai_config.json') as f:
        #     config_gen = json.load(f)
        #     model_name= config_gen["model"]["type"]
        tokenizer = AutoTokenizer.from_pretrained(model, token=False, use_fast=True, trust_remote_code=True)
        model = og.Model(model)
        # tokenizer = og.Tokenizer(model) # OGA Tokenizer not used since it does not support applying chat template
        device = model.device_type.lower()
    else:
        model_name = model.model_name
        tokenizer = model.tokenizer
        device = model.device    
    if device=="ryzenai":
        device = "cpu"
    print("Model: ", model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    TASK_NAME_MAPPING = {
        "stem": [
            # "abstract_algebra",
            #     "anatomy",
            "astronomy",
            #     "college_biology",
            #     "college_chemistry",
            #     "college_computer_science",
            #     "college_mathematics",
            #     "college_physics",
            #     "computer_security",
            #     "conceptual_physics",
            #     "electrical_engineering",
            #     "elementary_mathematics",
            #     "high_school_biology",
            #     "high_school_chemistry",
            #     "high_school_computer_science",
            #     "high_school_mathematics",
            #     "high_school_physics",
            #     "high_school_statistics",
            #     "machine_learning",
        ],
        "Humanities": [
        #     "formal_logic",
        #     "high_school_european_history",
        #     "high_school_us_history",
        #     "high_school_world_history",
        #     "international_law",
        #     "jurisprudence",
        #     "logical_fallacies",
        #     "moral_disputes",
        #     "moral_scenarios",
            "philosophy",
        #     "prehistory",
        #     "professional_law",
        #     "world_religions",
        ],
        "other": [
        #     "business_ethics",
        #     "college_medicine",
        #     "human_aging",
             "management",
        #     "marketing",
        #     "medical_genetics",
        #     "miscellaneous",
        #     "nutrition",
        #     "professional_accounting",
        #     "professional_medicine",
        #     "virology",
        #     "global_facts",
        #     "clinical_knowledge",
        ],
        # "social": [
        # "econometrics",
        # "high_school_geography",
        # "high_school_government_and_politics",
        # "high_school_macroeconomics",
        # "high_school_microeconomics",
        # "high_school_psychology",
        # "human_sexuality",
        # "professional_psychology",
        # "public_relations",
        # "security_studies",
        # "sociology",
        # "us_foreign_policy",
        # ],
    }

    SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
    choices = ["A", "B", "C", "D"]

    print("Categories: ", [k for k in TASK_NAME_MAPPING.keys()])
    print("Subjects: ", SUBJECTS)

    def format_example(line, include_answer=True):
        example = "Question: " + line["question"]
        for choice in choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

        if include_answer:
            example += "\nAnswer: " + line["answer"] + "\n\n"
        else:
            example += "\nAnswer:"
        return example

    def format_example_chat(line):
        example = (
            "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question.\n\n"
            + line["question"]
            + "\n"
        )
        for choice in choices:
            example += f'{choice}. {line[f"{choice}"]}\n'
        return example

    def generate_few_shot_prompt(k, subject, dev_df):
        def format_subject(subject):
            l = subject.split("_")
            s = ""
            for entry in l:
                s += " " + entry
            return s.strip()

        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )

        if k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += format_example(
                dev_df.iloc[i, :],
                include_answer=True,
            )
        return prompt

    def get_logits(
        tokenizer, model, inputs: List[str], framework="pytorch", device="cpu"
    ):
        input_ids = tokenizer(inputs, padding="longest")["input_ids"]
        input_ids = torch.tensor(input_ids, device=device)

        if input_ids.shape[1] > max_input_len:
            input_ids = input_ids[:, input_ids.shape[1] - max_input_len + 1 :]
        tokens = {"input_ids": input_ids}
        if framework == "oga":
            params = og.GeneratorParams(model)
            if max_length is None:
                raise ValueError(
                    "mmlu(framework='oga') requires max_length "
                    "(typically context_length from genai_config.json)"
                )
            params.set_search_options(max_length=max_length)
            generator = og.Generator(model, params)
            generator.append_tokens(input_ids)
            outputs = generator.get_output("logits")
            # Model uses fixed_prompt_length, so the logits tensor is padded to
            # that length. We must read at the last *real* input position
            # (input_ids is left-aligned with right padding); reading [:, -1, :]
            # would return logits at a padding position and yield garbage.
            last_real_pos = input_ids.shape[1] - 1
            logits = torch.tensor(outputs)[:, last_real_pos, :]
        else:
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
            outputs = model(input_ids, attention_mask=attention_mask)["logits"]
            logits = outputs[:, -1, :]
        log_probs = torch.nn.functional.softmax(logits, dim=-1)
        return log_probs, {"tokens": tokens}

    def process_before_extraction(gen, choice_dict):
        # replace the choice by letter in the generated sentence
        # from longest one to shortest one
        for key, val in sorted(
            choice_dict.items(), key=lambda x: len(x[1]), reverse=True
        ):
            pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
            gen = pattern.sub(key, gen)
        return gen

    def extract_choice(gen, choice_list):
        # answer is A | choice is A | choose A
        res = re.search(
            r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
            gen,
        )

        # A is correct | A is right
        if res is None:
            res = re.search(
                r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
                gen,
            )

        # straight answer: A
        if res is None:
            res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

        # simply extract the first appearred letter
        if res is None:
            res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

        if res is None:
            return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
        return res.group(1)

    def extract_answer(response, row):
        gen = process_before_extraction(
            response, {choice: row[choice] for choice in choices}
        )
        pred = extract_choice(gen, [row[choice] for choice in choices])
        return pred

    @torch.no_grad()
    def eval_subject(
        model,
        tokenizer,
        model_name,
        subject_name,
        test_df,
        framework,
        device,
        k=5,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        batch_size=1,
        **kwargs,
    ):
        result = []
        score = []
        save_response = []
        if (
            ("chat" in model_name.lower())
            or ("DeepSeek" in model_name)
            or ("Qwen1.5-MoE-A2.7B" in model_name)
            ):
            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                question = format_example_chat(row)
                message = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ]
                text = tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )

                if framework == "oga":
                    # Use OGA tokenizer
                    og_tokenizer = og.Tokenizer(model)
                    input_ids_ = og_tokenizer.encode(text)

                    params = og.GeneratorParams(model)
                    params.set_search_options(max_length=3096, temperature=0.7, top_p=0.9)
                    generator = og.Generator(model, params)
                    generator.append_tokens(input_ids_)

                    output_ids = []
                    while not generator.is_done():
                        generator.generate_next_token()
                        next_token = generator.get_next_tokens()[0]
                        output_ids.append(next_token)

                    generate_ids = output_ids
                    response = og_tokenizer.decode(generate_ids)
                    response = [response]  # to match Hugging Face response format

                else:
                    # HuggingFace flow
                    model_inputs = tokenizer([text], return_tensors="pt")
                    input_ids_ = model_inputs["input_ids"]
                    attention_mask = torch.ones(input_ids_.shape)
                    streamer = None  # Optional: TextStreamer(tokenizer)
                    generate_ids = model.generate(
                        input_ids=input_ids_,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        streamer=streamer,
                    )
                    response = tokenizer.batch_decode(
                        generate_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    
                # print(f"Decoded output: '{response}'")
                pred = extract_answer(response[0], row)
                save_response.append(response)

                if "answer" in row:
                    correct = 1 if pred == row["answer"] else 0
                    score.append(correct)
                result.append(pred)
        else:
            few_shot_prompt = (
                generate_few_shot_prompt(k, subject_name, dev_df) if few_shot else []
            )
            all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}
          
            if "llama" in model_name.lower() or "phi" in model_name.lower():
                choices = ["A", "B", "C", "D"]
                choices_ids = torch.tensor([
                    tokenizer(f" {c}")["input_ids"][-1] for c in choices
                ]).unsqueeze(0).to(device)
            else:
                choices = ["A", "B", "C", "D"]
                choices_ids = torch.tensor([
                    tokenizer(f" {c}")["input_ids"][-1] for c in choices
                ]).unsqueeze(0).to(device)


            idx_list = list(range(0, len(test_df), batch_size))
            for i in tqdm(idx_list):
                full_prompt_list = []
                answer_list = []
                for row in test_df.iloc[i : i + batch_size].to_dict(orient="records"):
                    question = format_example(row, include_answer=False)
                    full_prompt = few_shot_prompt + question
                    full_prompt_list.append(full_prompt)
                    if "answer" in row:
                        answer_list.append(row["answer"])

                logits, input_info = get_logits(
                    tokenizer, model, full_prompt_list, framework, device
                )
                softval = logits.gather(
                    1, choices_ids.expand(logits.size(0), -1)
                ).softmax(1)
                if softval.dtype in {torch.bfloat16, torch.float16}:
                    softval = softval.to(dtype=torch.float32)
                probs = softval.detach().cpu().numpy()
                
                for i in range(len(probs)):
                    for j, choice in enumerate(choices):
                        all_probs[f"prob_{choice}"].append(probs[i][j])
                    
                    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs[i])]

                    if answer_list != []:
                        correct = 1 if pred == answer_list[i] else 0
                        score.append(correct)
                       
                    result.append(pred)

        if save_result_dir:
            test_df["model_output"] = result
            if (
                ("chat" in model_name.lower())
                or ("DeepSeek" in model_name)
                or ("Qwen1.5-MoE-A2.7B" in model_name)
            ):
                test_df["model_response"] = save_response
            else:
                for i, choice in enumerate(choices):
                    test_df[f"prob_{choice}"] = all_probs[f"prob_{choice}"]
            if score:
                test_df["correctness"] = score
            os.makedirs(save_result_dir, exist_ok=True)
            test_df.to_csv(
                os.path.join(save_result_dir, f"{subject_name}_result.csv"),
                encoding="utf-8",
                index=False,
            )

        return score

    def cal_mmlu(res):
        acc_sum_dict = dict()
        acc_norm_sum_dict = dict()
        cnt_dict = dict()
        acc_sum = 0.0
        cnt = 0
        hard_cnt = 0
        hard_acc_sum = 0.0

        for class_ in TASK_NAME_MAPPING.keys():
            acc_sum_dict[class_] = 0.0
            acc_norm_sum_dict[class_] = 0.0
            cnt_dict[class_] = 0.0

            for tt in TASK_NAME_MAPPING[class_]:
                acc_sum += sum(res[tt])
                cnt += len(res[tt])

                acc_sum_dict[class_] += sum(res[tt])
                cnt_dict[class_] += len(res[tt])

        print("\n\n\n", "total cnt:", cnt, "\n")
        for k in TASK_NAME_MAPPING.keys():
            if k in cnt_dict:
                print("%s ACC: %.2f " % (k, acc_sum_dict[k] / cnt_dict[k] * 100))
        print("AVERAGE ACC:%.2f " % (acc_sum / cnt * 100))

    dev_result = {}
    for subject_name in tqdm(SUBJECTS):
        # val_file_path = os.path.join(args.eval_data_path, 'val', f'{subject_name}_val.csv')
        dev_file_path = os.path.join(eval_data_path, "dev", f"{subject_name}_dev.csv")
        test_file_path = os.path.join(
            eval_data_path, "test", f"{subject_name}_test.csv"
        )
        # val_df = pd.read_csv(val_file_path, names=['question','A','B','C','D','answer'])
        dev_df = pd.read_csv(
            dev_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )
        

        ### Only test nsamples entires for fast eval. est 2:30 hrs for one subject.
        # test_df = test_df.head(nsamples)

        score = eval_subject(
            model,
            tokenizer,
            model_name,
            subject_name,
            test_df,
            framework,
            device,
            dev_df=dev_df,
            k=5,
            few_shot=True,
            save_result_dir=f"outs/mmlu_eval_result",
            batch_size=1,
        )
        dev_result[subject_name] = score
    cal_mmlu(dev_result)