
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


###########################################################################

def template_pp(
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

            query_items = [last_resp]
            images = user_data['image']

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')

def get_sft_data_prompt(
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

            query_items = [user_data['messages'][0]['content']]

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')

def get_sft_data_image_and_prompt(
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

            if 'images' in user_data:
                images = user_data['images']
            query_items = [user_data['messages'][0]['content']]

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')

def get_sft_data_prompt_and_response(
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

            query_items = [user_data['messages'][0]['content'], user_data['messages'][1]['content']]

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')


def get_sft_data_image_and_prompt_and_response(
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

            if 'images' in user_data:
                images = user_data['images']
            query_items = [user_data['messages'][0]['content'], user_data['messages'][1]['content']]

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                
            f.write(json.dumps(data) + '\n')

def get_response(
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
            query_items = [last_resp]
            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def get_response_and_images(
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
            query_items = [last_resp]
            if 'images' not in user_data and 'image' in user_data:
                user_data['images'] = [user_data['image']]
            images = user_data['images']
            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def get_instruction(
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
            query_items = [user_data['instruction']]
            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')


def get_instruction_and_images(
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
            query_items = [user_data['instruction']]
            if 'images' not in user_data and 'image' in user_data:
                user_data['images'] = [user_data['image']]
            images = user_data['images']
            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def get_instruction_and_response(
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