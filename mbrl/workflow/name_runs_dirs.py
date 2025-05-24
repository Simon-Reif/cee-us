import os
import wandb

def get_wandb_name(params, run=None):
    pass

def get_working_dir(params, run):
    prefix = params.work_dir_prefix
    run_id = run.id
    data_tag = params.train_data_tag if hasattr(params, 'train_data_tag') else ''
    working_dir = os.path.join(prefix, data_tag, run_id)
    return working_dir
    
