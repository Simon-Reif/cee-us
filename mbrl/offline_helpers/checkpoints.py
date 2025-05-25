import json
import os
import numpy as np
import smart_settings

from mbrl.controllers.fb import ForwardBackwardController
# from mbrl.controllers.fb_cpr import FBcprController


def _get_cp_dir(working_dir, iter):
    return os.path.join(working_dir, f"checkpoint_{iter}")

# eval_return_dict contains:
#   "success_rates_eps"
#   "success_rates"
#   "z_rs"
#   "goals"
def save_fb_checkpoint(working_dir, iter, controller=None, eval_return_dict=None , loud=0):
    cp_dir = _get_cp_dir(working_dir, iter)
    os.makedirs(cp_dir, exist_ok=True)
    if controller is not None:    
        controller.save(cp_dir)
    if "success_rates" in eval_return_dict:
        success_rates_dict = eval_return_dict["success_rates"]
        np.save(os.path.join(cp_dir, "success_rate_dict.npy"), success_rates_dict)
    if "z_rs" in eval_return_dict:
        z_rs = eval_return_dict["z_rs"]
        np.save(os.path.join(cp_dir, "z_r_dict.npy"), z_rs)
    if "goals" in eval_return_dict:
        goals = eval_return_dict["goals"]
        np.save(os.path.join(cp_dir, "goals.npy"), goals)
    # TODO: look up how rollout buffers are saved
    # if rollout_buffer_dict is not None:
    #     np
    if loud > 0:
        print(f"Saved Checkpoint {iter}")
        if loud > 1:
            print(f"Average success rates:")
            for key, item in success_rates_dict.items():
                print(f"Task {key}: mean success {np.mean(item)}")


# dict of task: rollouts
def get_success_rates_dict(working_dir, iter):
    success_rates = np.load(os.path.join(_get_cp_dir(working_dir, iter), "success_rate_dict.npy"))
    return success_rates.item()

def get_zr_dict(working_dir, iter=None):
    if iter is not None:
        path = _get_cp_dir(working_dir, iter)
    else:
        path = working_dir
    zrs = np.load(os.path.join(path, "z_r_dict.npy"), allow_pickle=True)
    return zrs.item()

def get_goals_dict(working_dir, iter=None):
    if iter is not None:
        path = _get_cp_dir(working_dir, iter)
    else:
        path = working_dir
    goals = np.load(os.path.join(path, "goals.npy"), allow_pickle=True)
    return goals.item()

# B(s_next) for B at iter and the offline training buffer
def get_bs(working_dir, iter):
    return np.load(os.path.join(_get_cp_dir(working_dir, iter), "bs.npy"), allow_pickle=True)

# we assume parameters are saved in working_dir/
def get_fb_controller(working_dir=None, iter=None, cp_dir=None, cpr=False):
    if cp_dir is not None:
        working_dir = os.path.dirname(cp_dir)
    elif working_dir is not None and iter is not None:
        cp_dir = _get_cp_dir(working_dir, iter)
    else:
        raise ValueError("Either cp_dir or working_dir and iter must be provided")

    params = smart_settings.load(os.path.join(working_dir, 'settings.json'), make_immutable=False)
    # if cpr:
        # return FBcprController.load(cp_dir, params.controller_params)
    # else:
    return ForwardBackwardController.load(cp_dir, params.controller_params)

def mean_success_rates_to_csv(working_dir, iter):
    pass


def get_latest_checkpoint(working_dir, cpr=False):
    meta = smart_settings.load(os.path.join(working_dir, "meta.json"))
    agent = get_fb_controller(working_dir=working_dir, iter=meta["latest_checkpoint"], cpr=cpr)
    return agent, meta["latest_checkpoint"]

def save_meta(working_dir, iter):
    meta={}
    meta["latest_checkpoint"] = iter
    filename = os.path.join(working_dir, "meta.json")
    with open(filename, "w") as file:
        file.write(json.dumps(meta, sort_keys=True, indent=4))

# TODO: loading parts of FB controller only: here or in controller class?
# TODO: get latest checkpoint?