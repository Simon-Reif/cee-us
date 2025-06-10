import os
import numpy as np
import smart_settings
import wandb

from mbrl.workflow.config_helpers import val_from_dot_string

def _only_exp(number, exp_digits=1):
        exp_repr = np.format_float_scientific(number, exp_digits=exp_digits)
        n_list = exp_repr.split('e')
        return n_list[-1]
def _percentage_only(number):
        perc_repr = f"{number:2.0%}"
        return perc_repr.split('%')[0]

# for display of params in wandb names
param_name_map = {
    "disc": ("controller_params.train.discount", lambda x: _percentage_only(x)),
    "zdim": ("controller_params.model.archi.z_dim", lambda x: f"{x}"),
    # mr for mixing ratio
    "mr": ("controller_params.train.train_goal_ratio", lambda x: f"{float(x):1.2}"),
    "tau": ("controller_params.train.fb_target_tau", lambda x:_only_exp(x)),
    "orth": ("controller_params.train.ortho_coef", lambda x: f"{x}"),
    "exp_weight": ("training_data.planner_throw.weight", lambda x: f"{x}"),
    "exp_eps": ("training_data.planner_throw.max_episodes", lambda x: f"{x}"),
    "lr_actor": ("controller_params.train.lr_actor", lambda x: _only_exp(x)),
}

def _name_substr(params, tag):
    keys_dot_notation = param_name_map[tag][0]
    param_val = val_from_dot_string(params, keys_dot_notation)
    format_fun = param_name_map[tag][1]
    name_substr= f"{tag}{format_fun(param_val)}"
    return name_substr


def get_wandb_name(params, run=None):
    name_parts = []
    if hasattr(params.logging, 'wandb_name_prefix'):
        name_parts.append(params.logging.wandb_name_prefix)
    if hasattr(params.logging, "data_tag_in_name") and params.logging.data_tag_in_name and hasattr(params, 'train_data_tag'):
         name_parts.append(params.train_data_tag)
    if hasattr(params.logging, 'wandb_name_params'):
        wandb_name_params = params.logging.wandb_name_params
        params_substr = [_name_substr(params, tag) for tag in wandb_name_params]
        params_substr = " ".join(params_substr)
        name_parts.append(params_substr)
    name = " ".join(name_parts)
    return name
        


def get_working_dir(params, run):
    prefix = params.work_dir_prefix
    data_tag = params.train_data_tag if hasattr(params, 'train_data_tag') else ''
    run_id = run.id
    working_dir = os.path.join(prefix, data_tag, run_id)
    return working_dir
    

if __name__ == "__main__":
    params = smart_settings.load("experiments/cee_us/settings/common/fb/test_config.yaml")
    output_get_wandb_name = get_wandb_name(params)
    print(f"Output of get_wandb_name: {output_get_wandb_name}")