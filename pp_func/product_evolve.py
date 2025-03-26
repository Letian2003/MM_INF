
import os
from utils import load_data
import json
from typing import Union

def get_last_resp(
    data,
    step_name: str,
):
    # get the last response (from 'step_name' step, namely this step)
    if step_name in data['history'] and 'response' in data['history'][step_name]:
        last_resp = data['history'][step_name]['response']
    else:
        last_resp = None
    return last_resp

def first_process(
    data,
    idx,
):
    # automatically generate id if not exist
    if 'id' not in data or data['id'] in ['', None]:
        # 12 digits with leading zero
        data['id'] = f'{idx:012d}'
    # put all original data into 'history'
    if 'history' not in data:
        data = {'history': {}, 'user_data': data}
    return data


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

            query_items = [last_resp, user_data['response']]

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')

# def re_respond(
#     step_name: str,
#     input_file: str,
#     output_file: str,
# ):
#     dataset = load_data(input_file)
#     with open(output_file, 'w') as f:
#         for i, data in enumerate(dataset):
#             data = first_process(data, i)
#             last_resp = get_last_resp(data, step_name)
#             history, user_data  = data['history'], data['user_data']
#             query_items, images, videos, audios = [], [], [], []
#             ##### get ['query_items', 'images', 'videos', 'audios']

#             user_data['instruction'] = history['optimize']['response']
#             user_data['response'] = last_resp
#             query_items = [user_data['instruction'], user_data['response']]

#             ############################################################
#             data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
#             f.write(json.dumps(data) + '\n')