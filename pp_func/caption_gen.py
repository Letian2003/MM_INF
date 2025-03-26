
import os
from utils import load_data
import json
from typing import Union
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

            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

def caption_pp(
    step_name: str,
    input_file: str,
    output_file: str,
):

    dataset = load_data(input_file)
    final_filename = os.path.dirname(output_file) + '/final_caption_gen.jsonl'
    with open(output_file, 'w') as f, open(final_filename, 'w') as f_out:
        for data in dataset:
            last_resp = get_last_resp(data, step_name)
            history, user_data  = data['history'], data['user_data']
            query_items, images, videos, audios = [], [], [], []
            ##### get ['query_items', 'images', 'videos', 'audios'] #####
            user_data['caption'] = last_resp
            user_data['query'] = '<image>\nPlease describe this image in detail.'
            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')

            new_data = {}
            new_data['image'] = user_data['image']
            new_data['caption'] = last_resp
            new_data['query'] = user_data['query']
            f_out.write(json.dumps(new_data) + '\n')