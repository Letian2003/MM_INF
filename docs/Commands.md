## Commands  
- `--config_file`: Path to the configuration file.  
- `--enable_history` (default, optional) / `--no-enable_history`: Save the complete history. If disabled, only the response from the current step will be saved.  
- `--input_file` (optional): Path to the input file. If specified, it will override the input file path in the config.  
- `--save_dir` (optional): Path to the save directory. If specified, it will override the save path in the config.  

### Example

```bash
python3 infer_pipeline.py \
    --config_file config/oasis.yaml \
    --no-enable_history
```