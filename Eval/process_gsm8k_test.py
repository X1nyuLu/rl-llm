import json
import os
from datasets import load_dataset

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions_for_eval(split = "train"):
    data = load_dataset('gsm8k', data_dir='main')[split] # type: ignore
    lst_data = []
    for i, item in enumerate(data):
        dct_data = {}
        dct_data['id'] = i
        dct_data['question'] = item['question']
        dct_data['answer'] = extract_hash_answer(item['answer'])
        lst_data.append(dct_data)
    return lst_data

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


if __name__ == "__main__":
    lst_test_data = get_gsm8k_questions_for_eval(split="test")
    save_json_or_jsonl(lst_test_data, "Eval/gsm8k_test.json")

