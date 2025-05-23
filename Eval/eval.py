import argparse
import json
import os
import random
import re

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoProcessor

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def clean_number(text: str) -> str:
    """Remove commas from number strings."""
    return text.replace(",", "")


def load_json_or_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        print(f"Loaded from {file_path}")
        return data
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def save_json_or_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                for d in data:
                    f.write(json.dumps(d, ensure_ascii=False) + '\n')
            else:
                json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Saved to {file_path}")
    except Exception as e:
        print(f"Failed to save {file_path}: {e}")



def make_inference(lst_data):
    random.seed(args.seed)
    inputs = []
    for data in lst_data:
        messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": data["question"]
                },
            ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs.append(prompt)
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params,
        lora_request=adapter,
    )
    for data, output in zip(lst_data, outputs):
        data['response'] = output.outputs[0].text
        pred = extract_xml_answer(data['response'])
        pred = re.search(r'\d+', pred).group()
        data['final_answer'] = pred
    return lst_data








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=None, help="base model path")
    parser.add_argument("--lora-adapter", type=str, default=None, help="lora adapter path")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="dataset path")
    parser.add_argument("--out-dir", type=str, default="output", help="output directory")
    parser.add_argument("--infer-only", action="store_true", help="only infer, no evaluation")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    assert args.base_model is not None, "base-model is required"
    assert args.lora_adapter is not None, "lora-adapter is required"

    llm = LLM(model=args.base_model, enable_lora=True, max_lora_rank=32)
    # llm.load_lora_weights(args.lora_adapter, adapter_name="default")
    sampling_params = SamplingParams(
    temperature=0,
    max_tokens=512)
    processor = AutoProcessor.from_pretrained(args.base_model)
    
    adapter = LoRARequest("default", 1, args.lora_adapter)
    
    lst_data = load_json_or_jsonl(args.dataset)
    
    infer_results = make_inference(lst_data)

    full_filename = os.path.basename(args.base_model)  # "example.txt"
    filename = os.path.splitext(full_filename)[0]
    full_filename = os.path.basename(args.lora_adapter)  # "example.txt"
    filename += "_" + os.path.splitext(full_filename)[0]
    full_filename = os.path.basename(args.dataset)  # "example.txt"
    filename += "_" + os.path.splitext(full_filename)[0]

    if args.infer_only:
        save_json_or_jsonl(infer_results, os.path.join(args.out_dir, f"{filename}_result.json"))
    else:
        correct_count = 0
        for data in infer_results:
            gt = clean_number(data['answer'])
            pred = clean_number(data['final_answer'])
            if int(pred) == int(gt):
                correct_count += 1

        print(f"Correct count: {correct_count}")
        print(f"Accuracy: {correct_count / len(infer_results)}")
    