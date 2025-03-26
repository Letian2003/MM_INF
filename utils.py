import json
import re


def load_data(
    input_file: str,
):
    if input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError(f'input_file must be json or jsonl, but got {input_file}')
    return data


def extract_score(text):
    pattern = r"\d+"
    matches = list(re.finditer(pattern, text))
    # print(matches)
    if matches:
        last_match = matches[-1]
        try:
            score = int(last_match.group())
            return score if 0 <= score <= 10 else -1
        except:
            return -1
    return -1