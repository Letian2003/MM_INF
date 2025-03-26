
import os
from utils import load_data, extract_score
import json
from typing import Union
from pp_func.template_pp import first_process, get_last_resp


def get_sft_data_instruction_and_response(
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

            query_items = [user_data['instruction'],user_data['response']]
            if 'images' not in user_data and 'image' in user_data:
                user_data['images'] = [user_data['image']]

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')


def score_pp(
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

            query_items = [user_data['instruction'],user_data['response']]
            images = user_data['images']

            user_data['ori_score'] = extract_score(last_resp)

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')

def feedback_pp(
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

            query_items = [user_data['instruction'], last_resp]

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')

def optimize_pp(
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
            ##### get ['query_items', 'images', 'videos', 'audios']

            query_items = [last_resp]
            images = user_data['images']
            user_data['new_instruction'] = last_resp

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')

def re_respond(
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
            ##### get ['query_items', 'images', 'videos', 'audios']
            
            user_data['new_response'] = last_resp
            query_items = [user_data['new_instruction'], user_data['new_response']]

            ############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')

def score_2_pp(
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

            user_data['new_score'] = extract_score(last_resp)

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')