import os
# silent logging info
os.environ['NCCL_DEBUG'] = "WARN"
os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
import sys
import torch
import ray
import importlib
import json
import multiprocessing
import argparse
import yaml
import contextlib
import gc
from swift.llm import InferRequest, RequestConfig, VllmEngine
from utils import load_data
import time
try:
    torch.multiprocessing.set_start_method('spawn')
except:
    pass
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from oasis.hack_VllmEngine import VllmEngine
sys.modules["swift.llm"].VllmEngine = VllmEngine

class MMINFDataSynthesisPipeline:
    def __init__(self):
        self.engine = None
        self.model_id_or_path = None

    def destroy_model(self):
        pass

    def update_model(self, model_id_or_path, max_model_len=32768, limit_mm_per_prompt={'image':5, 'video':2}, batch_size=128):
        """
        更新模型
        :param model_id_or_path: 模型的路径或ID
        :param max_model_len: 模型的最大长度
        :param limit_mm_per_prompt: 每个提示的多模态限制
        :param batch_size: 批次大小
        """
        if self.model_id_or_path == model_id_or_path:
            return
        self.model_id_or_path = model_id_or_path
        if self.engine:
            self.destroy_model()
        
        world_size = torch.cuda.device_count()
        self.engine = VllmEngine(
            model_id_or_path,
            max_model_len=max_model_len,
            limit_mm_per_prompt=limit_mm_per_prompt,
            tensor_parallel_size=world_size,
            max_num_seqs=batch_size,
            gpu_memory_utilization=0.8,
        )

    def format_prompt(self, prompt_raw, query_items):
        try:
            # 如果prompt定义在python文件中，则使用importlib导入
            module_name, func_name = prompt_raw.rsplit(".", 1)
            module = importlib.import_module(module_name)
            prompt_func = getattr(module, func_name)
            prompt = prompt_func(*query_items)
        except:
            # 如果prompt是字符串，则直接使用format方法，并检查{}占位符的数量是否和query_items一致
            # 预期prompt里的{}占位符数量和query_items相等，才可以正常生成prompt
            assert prompt_raw.replace("{{", "").replace("}}", "").count("{}") == len(query_items), f"Number of placeholders ({prompt_raw.count('{}')}) does not match the number of query items ({len(query_items)})"
            prompt = prompt_raw.format(*query_items)
        
        return prompt

    def prepare_data(self, data, prompt_raw, sys_prompt):
        """
        准备推理请求数据
        :param data: 输入数据
        :param prompt_raw: 原始提示
        :return: 推理请求对象
        """
        query_items = data['query_items']
        images = data.get('images', [])
        audios = data.get('audios', [])
        videos = data.get('videos', [])
        prompt = self.format_prompt(prompt_raw, query_items)

        request_message = {
            'role': 'user',
            'content': []
        }


        if '<image>' in prompt:
            num_images = prompt.count('<image>')
            if num_images != len(images):
                raise ValueError(f'Number of images ({len(images)}) does not match the number of <image> placeholders ({num_images}) in the prompt.')
            request_message['content'] = prompt
            return InferRequest(messages=[request_message], images = images)
        else:
            for image in images:
                request_message['content'].append(
                    {
                        'type': 'image',
                        'image': image
                    }
                )
            for audio in audios:
                request_message['content'].append(
                    {
                        'type': 'audio',
                        'audio': audio
                    }
                )
            for video in videos:
                request_message['content'].append(
                    {
                        'type': 'video',
                        'video': video
                    }
                )
            request_message['content'].append(
                {
                    'type': 'text',
                    'text': prompt
                }
            )
            return InferRequest(messages=[request_message])

    def infer(self, step_name, model_id_or_path, input_file, output_file, save_dir, prompt_func, sys_prompt=None, enable_history=True, oasis=False, batch_size=128):
        """
        进行推理
        :param step_name: 步骤名称
        :param model_id_or_path: 模型的路径或ID
        :param input_file: 输入文件
        :param output_file: 输出文件
        :param save_dir: 保存目录
        :param prompt_func: 提示函数或字符串
        :param oasis: 是否使用oasis
        :param batch_size: 批次大小
        """
        input_data = load_data(input_file)

        # 检查是否有推理结果的缓存，如果有，则从缓存中读取，只推理没有推理结果的数据
        cache_dir = os.path.join(save_dir, 'cache')
        cache_file = os.path.join(cache_dir, 'cache.jsonl')
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(cache_file) or len(load_data(cache_file)) == 0:
            with open(cache_file, 'w') as f:
                json.dump({}, f)
        cache_data = load_data(cache_file)[0]
        if step_name not in cache_data:
            cache_data[step_name] = 0
        processed_num = cache_data[step_name]
        input_data = input_data[processed_num:]
        if not input_data:
            return
        
        # 检查是否需要切换模型
        self.update_model(model_id_or_path, batch_size=batch_size)
        
        # 创建推理请求
        infer_requests = []
        request_config = RequestConfig(max_tokens=1024, temperature=0)
        if oasis:
            request_config.encode_hack = True
            request_config.hack_idx = -5
            request_config.temperature = 1.0
            request_config.top_k = 20
            request_config.top_p = 1.0
            request_config.min_tokens = 2
            request_config.repetition_penalty = 1.0
            request_config.max_tokens = 512
            request_config.top_logprobs = 10

        if not os.path.exists(output_file) or processed_num == 0:
            with open(output_file, 'w') as f:
                pass
        
        for data in input_data:
            infer_requests.append(self.prepare_data(data, prompt_func, sys_prompt=sys_prompt))

        def post_process(step_name, prompt_func, data, resp, enable_history):
            if 'history' not in data:
                data['history'] = {}
            if not enable_history:
                data['history'] = {}
                data['history'][step_name] = {
                    'response': resp
                }
            else:
                data['history'][step_name] = {
                    'prompt_raw': prompt_func,
                    'query_items': data['query_items'],
                    'response': resp
                }
                
                if 'images' in data:
                    data['history'][step_name]['images'] = data['images']
                if 'videos' in data:
                    data['history'][step_name]['videos'] = data['videos']
                if 'audios' in data:
                    data['history'][step_name]['audios'] = data['audios']
                data = {'history': data['history'], 'user_data': data['user_data']}
            return data

        with open(output_file, 'a') as f:
            for i in range(0, len(infer_requests), batch_size):
                # 获取当前批次的推理请求
                batch_infer_requests = infer_requests[i:i + batch_size]
                resp_list = []
                resp_list = self.engine.infer(batch_infer_requests, request_config)
                resp_list = [resp.choices[0].message.content for resp in resp_list]
                for data, resp in zip(input_data[i:i + batch_size], resp_list):
                    # 保存推理结果到history中
                    data = post_process(step_name, prompt_func, data, resp, enable_history)
                    f.write(json.dumps(data) + '\n')
                cache_data[step_name] += len(batch_infer_requests)
                with open(cache_file, 'w') as f_cache:
                    json.dump(cache_data, f_cache)

def exec_group(config_list, input_file, save_dir, enable_history, filename_queue):
    mminf_pipeline = MMINFDataSynthesisPipeline()

    for config in config_list:
        step_name = config.get('step_name')
        exec_step(config, mminf_pipeline, input_file, save_dir, enable_history, step_name, filename_queue)
        input_file = filename_queue.get()
    filename_queue.put(os.path.basename(input_file))

def exec_step(config, mminf_pipeline, input_file, save_dir, enable_history, step_name, filename_queue):
    """
    执行一个步骤
    :param config: 步骤配置
    :param input_file: 输入文件
    :param save_dir: 保存目录
    :param step_name: 步骤名称
    :param filename_queue: 文件名队列
    """
    
    # 获取这个step的模型checkpoint路径
    model_id_or_path = config.get('model_id_or_path')
    # 获取这个step的后处理函数
    pp_func_list = config.get('pp_func', [])
    if pp_func_list is None:
        pp_func_list = []
    if isinstance(pp_func_list, str):
        pp_func_list = [pp_func_list]

    if not os.path.exists(input_file):
        input_file = os.path.join(save_dir, input_file)

    if model_id_or_path: # 如果有模型，就做推理，否则跳过推理做后处理
        prompt = config['prompt']
        sys_prompt = config.get('sys_prompt', None)
        output_file = os.path.join(save_dir, step_name + '_mid.jsonl')
        oasis = config.get('oasis', False)
        if oasis:
            print('********** [INFO] Use Oasis **********')

        mminf_pipeline.infer(
            step_name,
            model_id_or_path,
            input_file,
            output_file,
            save_dir,
            sys_prompt = sys_prompt,
            prompt_func=prompt,
            enable_history = enable_history,
            oasis=oasis
        )

        input_file = output_file

    for pp_func in pp_func_list:
        # 定位后处理python函数的路径&导入函数
        module_name, func_name = pp_func.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        output_file = os.path.join(save_dir, step_name + '_post.jsonl')
        # 执行后处理函数
        func(
            step_name,
            input_file,
            output_file
        )

        input_file = output_file

    filename_queue.put(os.path.basename(input_file))

def parse_args():
    """
    解析命令行参数
    :return: 解析后的参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--enable_history', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config_file = args.config_file
    enable_history = args.enable_history
    input_file = args.input_file
    save_dir = args.save_dir
    print('enable_history:', enable_history)
    with open(config_file, "r", encoding="utf-8") as file:
        running_config = yaml.safe_load(file)

    top_level_keys = list(running_config.keys())

    if save_dir is None:
        save_dir = running_config['base_config']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    group_list = []
    now_group = []
    last_model = None

    for step_key in top_level_keys[1:]:
        running_config[step_key]['step_name'] = step_key
        cur_model = running_config[step_key].get('model_id_or_path')
        if cur_model is not None and last_model is not None and cur_model != last_model:
            group_list.append(now_group)
            now_group = []
            last_model = None
        now_group.append(running_config[step_key])
        if last_model is None:
            last_model = cur_model
    if now_group:
        group_list.append(now_group)

    for group in group_list:
        # NOTE: cannot delete vllm and release gpu memory in one process, 
        #       so we use multiprocessing to run each step
        # FIXME: how to release gpu memory in one process?
        filename_queue = multiprocessing.Queue()

        if input_file is None:
            input_file = running_config['base_config']['input_file']

        p = multiprocessing.Process(target=exec_group, args=(group, input_file, save_dir, enable_history, filename_queue))
        p.start()
        p.join()

        input_file = filename_queue.get()
