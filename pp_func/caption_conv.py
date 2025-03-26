
import os
from utils import load_data
import json
from typing import Union

from pp_func.template_pp import first_process, get_last_resp

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

            user_data['images'] = [user_data['image']]
            images = user_data['images']

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def caption_pp(
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
            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def question_generation_pp(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    with open(output_file, 'w') as f:
        for data in dataset:
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']
            # query_items, images, videos, audios = [], [], [], []
            data['images'] = user_data['images']

            if len(last_resp.split('\n')) >= 8:
                continue
            for question in last_resp.split('\n'):
                if question == '':
                    continue
                user_data['instruction'] = question
                data['query_items'] = [question]
                f.write(json.dumps(data) + '\n')



def respond_pp(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    with open(output_file, 'w') as f:
        for data in dataset:
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']
            ##### get ['query_items', 'images', 'videos', 'audios'] #####
            user_data['response'] = last_resp
            #############################################################
            f.write(json.dumps(data) + '\n')