
## Custom tasks
We have provided several example tasks in the `config/` directory. You can use them as a reference to create your own task.
To create a new task, you need to create a new config file under the `config/` directory, and prepare post-processing functions under the `pp_func/` directory.


### Config  

The structure of the config file is as follows:  
```plaintext
config.yaml
│── base_config
│   ├── input_file : json / jsonl
│   ├── save_dir
│── step1_name
│   ├── model_id_or_path
│   ├── prompt
│   ├── pp_func
│   ├── args
│── step2_name
│   ├── ...
```

- `model_id_or_path`: If no model is used in this step, set it to `null`
- `prompt`: If a model is used in this step, specify the prompt. It can include "{}" as a format placeholder. It can also be a function path with the function defined under `prompt_func/`. The parameters required for both approaches are defined in `pp_func`.
    - eg1 : `"Hello, {}"`
    - eg2 : `"prompt_func.oasis_prompt.extract_query_prompt_new"`
- `pp_func`: Post-processing function specified as a function path. The function should be defined under `pp_func/`. It can be null, a string, or a list.
    - eg1 : `"pp_func.caption_gen.caption_pp"`
    - eg2 : `["pp_func.caption_gen.caption1", "pp_func.caption_gen.caption2"]`


### File Structure
Intermediate data will be saved in the specified `save_dir` with the name of the corresponding step.
- Inference results will be stored in `step_mid.jsonl`.
- Post-processed results will be stored in `step_post.jsonl`.


Data Format
```json
{
    "history": {
        "step1":{...},
    },
    "user_data": {"images":[...]},
    "query_items": [], "images": [], ...
}
```
- `history`: Historical query data, updated with each inference.
- `user_data`: User data that the user can update themselves.
- `query_items`: A list of parameters to be passed to the next prompt, which the user needs to fill in.
- `images` / `videos` / `audios`: Lists of multimodal content to be passed to the next prompt, which the user needs to provide.


### post_process function
Template:
```python
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
            user_data['images'] = [user_data['image']]
            images = user_data['images']
            #############################################################
            data['query_items'], data['images'], data['videos'], data['audios'] = query_items, images, videos, audios
            f.write(json.dumps(data) + '\n')
```
This function write `images` into `user_data` and uses the `images` as input for the next prompt.

In general, users need to provide `query_items`, `images`, `videos`, and `audios` for the next query based on the previous response. Additionally, users can update `user_data` as needed. We have prepared several standard post-processing functions under `pp_func/template_pp.py`.


## Example  
Using the caption_gen  as an example, here is the basic configuration:  
```yaml
base_config:
  input_file: ./input.jsonl
  save_dir: ./caption_gen
```

Input Data Format
```json
{
    "image": "xxx",
    ...
}
```

### step1: pre_process
config:
```yaml
pre_process:
  model_id_or_path: null
  pp_func: pp_func.caption_gen.pre_process
```
This step does not use a model; it directly reads the data and performs preprocessing.  
- In the first step, the program automatically places the input data into `user_data` and initializes `history`. Users need to provide the `images` required for generating captions in the next step.  
- The prompt for the next step is defined in the next step's config. Since no parameters need to be passed, `query_items` does not need to be filled in.  

The content for the post-processing function in this step is as follows:  
```python
user_data['images'] = [user_data['image']]
images = user_data['images']
```
Input file:
```json
{
    "history": {}, 
    "user_data": {"image": "./images/12.jpg", "id": "000000000000"}, 
    "query_items": [], 
    "images": ["./images/12.jpg"], 
    "videos": []
}
```

### step2: caption
config:
```yaml
caption:
  model_id_or_path: YOUR_PATH_TO/Qwen2-VL-72B-Instruct
  prompt: Please describe this image in detail.
  pp_func: pp_func.caption_gen.caption_pp
```

In this step, the provided image from the previous step is used with the prompt to generate a caption. After generation, the query content and results are automatically recorded in `history`:  
```json
"history": {
    "caption": 
    {
        "prompt_raw": "Please describe this image in detail.", "query_items": [], "response": "The image shows...", "images": ["./images/12.jpg"], "videos": []
    }
}
```
The post-processing function store the caption and query into `user_data`:
```python
user_data['caption'] = last_resp
user_data['query'] = '<image>\nPlease describe this image in detail.'
```

Full content of output file `caption_post.jsonl`:
```json
{
    "history": {
        "caption": 
        {
            "prompt_raw": "Please describe this image in detail.", "query_items": [], "response": "The image shows...", "images": ["./images/12.jpg"], "videos": []
        }
    },
    "user_data": {"image": "./images/12.jpg", "id": "000000000000", "caption": "The image shows...", "query": "<image>\nPlease describe this image in detail."}, 
    "query_items": [], "images": [], "videos": [],
}
```