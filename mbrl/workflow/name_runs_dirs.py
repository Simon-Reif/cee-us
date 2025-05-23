import os
import wandb

def get_wandb_name(params, run=None):
    pass

def get_working_dir(params, run):
    prefix = params.work_dir_prefix
    #TODO: enable longer names
    run_id = run.id
    working_dir = os.path.join(prefix, run_id)
    return working_dir
    
