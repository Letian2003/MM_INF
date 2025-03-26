
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

            if 'image' in user_data:
                images = [user_data['image']]
            if 'query_items' in user_data:
                query_items = user_data['query_items']

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

def devide_pp(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    with open(output_file, 'w') as f:
        for data in dataset:
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']

            data['images'] = [user_data['image']]
            for para in last_resp.split('\n'):
                if para == '':
                    continue
                new_data = data
                new_data['query_items'] = [para]
                f.write(json.dumps(new_data) + '\n')


def merge(
    step_name: str,
    input_file: str,
    output_file: str,
):
    dataset = load_data(input_file)
    with open(output_file, 'w') as f:
        correct = 1
        for i, data in enumerate(dataset):
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']
            query_items, images, videos, audios = [], [], [], []
            ##### get ['query_items', 'images', 'videos', 'audios'] #####
            
            if last_resp != 'Correct':
                correct = 0

            if (i == len(dataset) - 1 or user_data['id'] != dataset[i+1]['user_data']['id']) and correct:
                user_data['caption'] = history['caption']['response']
                # data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
                f.write(json.dumps(data) + '\n')
                correct = 1
