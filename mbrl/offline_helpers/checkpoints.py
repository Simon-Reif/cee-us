import json
import os
import numpy as np
import smart_settings

from mbrl.controllers.fb import ForwardBackwardController


def _get_cp_dir(working_dir, iter):
    return os.path.join(working_dir, f"checkpoint_{iter}")

# success_reats: numpy array
# controller own save method
# z_rs, bs numpy arrays
def save_fb_checkpoint(working_dir, iter, success_rates_dict=None, controller:ForwardBackwardController=None, z_r_dict=None, bs=None, 
                       rollout_buffer_dict=None, loud=0):
    cp_dir = _get_cp_dir(working_dir, iter)
    os.makedirs(cp_dir, exist_ok=True)
    if success_rates_dict is not None:
        np.save(os.path.join(cp_dir, "success_rate_dict.npy"), success_rates_dict)
    if controller is not None:    
        controller.save(cp_dir)
    if z_r_dict is not None:
        np.save(os.path.join(cp_dir, "z_r_dict.npy"), z_r_dict)
    if bs is not None:
        np.save(os.path.join(cp_dir, "bs.npy"), bs)
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

# B(s_next) for B at iter and the offline training buffer
def get_bs(working_dir, iter):
    return np.load(os.path.join(_get_cp_dir(working_dir, iter), "bs.npy"), allow_pickle=True)

# we assume parameters are saved in working_dir/
def get_fb_controller(working_dir, iter):
    params = smart_settings.load(os.path.join(working_dir, 'settings.json'), make_immutable=False)
    return ForwardBackwardController.load(_get_cp_dir(working_dir, iter), params.controller_params)

def mean_success_rates_to_csv(working_dir, iter):
    pass


def get_latest_checkpoint(working_dir):
    meta = smart_settings.load(os.path.join(working_dir, "meta.json"))
    agent = get_fb_controller(working_dir, meta["latest_checkpoint"])
    return agent, meta["latest_checkpoint"]

def save_meta(working_dir, iter):
    meta={}
    meta["latest_checkpoint"] = iter
    filename = os.path.join(working_dir, "meta.json")
    with open(filename, "w") as file:
        file.write(json.dumps(meta, sort_keys=True, indent=4))

# TODO: loading parts of FB controller only: here or in controller class?
# TODO: get latest checkpoint?