
import os
from utils import load_data
import json
from typing import Union
import re
from pp_func.template_pp import first_process, get_last_resp

def pre_process(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    with open(output_file, 'w') as f:
        for i, data in enumerate(dataset):
            data = first_process(data, i)
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']
            query_items, images, videos, audios = [], [], [], []
            ##### get ['query_items', 'images', 'videos', 'audios'] #####

            images = [user_data['image']]
            user_data['images'] = [user_data['image']]

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')

def hooking_pp(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    with open(output_file, 'w') as f:
        for data in dataset:
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']
            query_items, images, videos, audios = [], [], [], []
            ##### get ['query_items', 'images', 'videos', 'audios'] #####

            query_items = [last_resp]
            # images = history['images']

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def inst_extract_pp(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    with open(output_file, 'w') as f:
        for data in dataset:
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']
            query_items, images, videos, audios = [], [], [], []
            ##### get ['query_items', 'images', 'videos', 'audios'] #####

            if last_resp.startswith('Instruction:'):
                instruction = last_resp.lstrip('Instruction:').strip()
            else:
                continue
            user_data['instruction'] = instruction

            query_items = [instruction]
            images = user_data['images']

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def extract_score(text):
    # print(text)
    # print()
    pattern = r"\[\[(.*?)\]\]"
    matches = list(re.finditer(pattern, text))
    match_last = matches[-1] if matches else None
    # print(match_last)
    if match_last:
        content = match_last.group(1)
        # print(content)
        try:
            score = int(content)
            return score if 1 <= score <= 5 else -1
        except:
            return -1
        
    pattern = r"\b\d+\b"
    matches = list(re.finditer(pattern, text))
    if matches:
        last_match = matches[-1]
        try:
            score = int(last_match.group())
            return score if 1 <= score <= 5 else -1
        except:
            return -1
    return -1

def solvable_pp(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    with open(output_file, 'w') as f:
        for data in dataset:
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']
            query_items, images, videos, audios = [], [], [], []
            ##### get ['query_items', 'images', 'videos', 'audios'] #####
            user_data['solvable_score'] = extract_score(last_resp)

            query_items = [user_data['instruction']]
            images = user_data['images']
            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def hallucination_pp(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    with open(output_file, 'w') as f:
        for data in dataset:
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']
            query_items, images, videos, audios = [], [], [], []
            ##### get ['query_items', 'images', 'videos', 'audios'] #####
            user_data['hallucination_score'] = extract_score(last_resp)

            query_items = [user_data['instruction']]
            images = user_data['images']
            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def clarity_pp(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    with open(output_file, 'w') as f:
        for data in dataset:
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']
            query_items, images, videos, audios = [], [], [], []
            ##### get ['query_items', 'images', 'videos', 'audios'] #####
            user_data['clarity_score'] = extract_score(last_resp)

            query_items = [user_data['instruction']]
            # images = history['images']
            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def nonsense_pp(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    with open(output_file, 'w') as f:
        for data in dataset:
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']
            query_items, images, videos, audios = [], [], [], []
            ##### get ['query_items', 'images', 'videos', 'audios'] #####
            user_data['nonsense_score'] = extract_score(last_resp)

            if user_data['solvable_score'] < 3 or user_data['hallucination_score'] != 5 or user_data['clarity_score'] < 3 or user_data['nonsense_score'] != 5 or user_data['solvable_score']+user_data['clarity_score'] < 7:
                continue

            query_items = [user_data['instruction']]
            images = user_data['images']

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def respond_pp(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    final_filename = os.path.dirname(output_file) + '/final_oasis.jsonl'
    with open(final_filename, 'w') as f_fin:
        with open(output_file, 'w') as f:
            for data in dataset:
                last_resp = get_last_resp(data, step_name)
                history, user_data  = data['history'], data['user_data']
                query_items, images, videos, audios = [], [], [], []
                ##### get ['query_items', 'images', 'videos', 'audios'] #####

                user_data['response'] = last_resp
                
                #############################################################
                data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                f.write(json.dumps(data) + '\n')

                new_data = {
                    'image': user_data['image'],
                    'conversations': [
                        {
                            'from': 'human',
                            'value': user_data['instruction'],
                        },
                        {
                            'from': 'gpt',
                            'value': user_data['response'],
                        },
                    ],
                    'id': user_data['id'],
                    'data_source': 'oasis',
                }
                f_fin.write(json.dumps(new_data) + '\n')