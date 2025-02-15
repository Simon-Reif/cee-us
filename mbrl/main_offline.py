import logging
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from mbrl import allogger, torch_helpers
from mbrl.controllers.fb import ForwardBackwardController
from mbrl.environments import env_from_string
from mbrl.helpers import gen_rollouts
from mbrl.params_utils import read_params_from_cmdline, save_settings_to_json
from mbrl.rollout_utils import RolloutManager
from mbrl.rolloutbuffer import RolloutBuffer
from mbrl.seeding import Seeding
from mbrl.offline_helpers.checkpoints import save_fb_checkpoint
from mbrl.offline_helpers.eval import eval, print_best_success_by_task, update_best_success_by_task


#TODO: option to continue training from a checkpoint
def main(params):
    logger = allogger.get_logger(scope="main", basic_logging_params={"level": logging.INFO})
    Seeding.set_seed(params.seed if "seed" in params else None)
    # update params file with seed (either given or generated above)
    params_copy = params._mutable_copy()
    params_copy["seed"] = Seeding.SEED
    
    #using params copy to use existing code defined on params with different tasks etc
    
    with open(os.path.join(params.training_data_dir, 'rollouts'), 'rb') as f:
        buffer = pickle.load(f)
    obs_dim=buffer[0]["observations"].shape[-1]
    action_dim=buffer[0]["actions"].shape[-1]
    
    params_copy.controller_params.model.obs_dim = obs_dim
    params_copy.controller_params.model.action_dim = action_dim
    fb_controller = ForwardBackwardController(params_copy.controller_params)
    #fb_controller = ForwardBackwardController(params.controller_params, obs_dim, act_dim)
    save_settings_to_json(params_copy, params.working_dir)

    # print("_______________Debugging_______________")
    # print(type(buffer))
    # print(buffer[0]["observations"].shape)
    # print("_______________Debugging_End_______________")

    debug=False
    best_success_by_task = defaultdict(dict)
    for iteration in tqdm(range(params.num_train_steps)):
    #for iteration in tqdm(range(10)):

        metrics = fb_controller.update(buffer, iteration)

        if iteration % params.log_every_updates == 0:
            for k, v, in metrics.items():
                logger.log(torch_helpers.to_numpy(v), key=k)

        ### Debug
        # if debug:
        #     params_copy["number_of_rollouts"] = 2
        ### Debug End
        if debug or (iteration % params.eval.eval_every_steps == 0 and iteration > 0):
            success_rates_dict, z_r_dict, bs = eval(fb_controller, buffer, params_copy, iteration)
            # TODO: move this into eval
            updated=update_best_success_by_task(best_success_by_task, success_rates_dict, iteration, debug)
            if updated:
                #TODO: for some reason this doesn't overwrite the file
                print_best_success_by_task(best_success_by_task, to_yaml=True, working_dir=params.working_dir)

            save_fb_checkpoint(params.working_dir, iteration, success_rates_dict=success_rates_dict, 
                               controller=fb_controller, z_r_dict=z_r_dict, bs=bs, loud=2 if debug else 0)
            
        
        allogger.get_root().flush(children=True)

    ###
    # End Main Loop
    ###

    #TODO save final model
    #TODO generate summary (plot) of best results
    #TODO maybe do sth with "total_metrics"
    print_best_success_by_task(best_success_by_task, console=True)
    allogger.close()

    return 0



if __name__ == "__main__":
    params = read_params_from_cmdline(verbose=True)

    os.makedirs(params.working_dir, exist_ok=True)

    allogger.basic_configure(
        logdir=params.working_dir,
        default_outputs=["tensorboard"],
        manual_flush=True,
        tensorboard_writer_params=dict(min_time_diff_btw_disc_writes=1),
    )

    allogger.utils.report_env(to_stdout=True)

    save_settings_to_json(params, params.working_dir)

    if "device" in params:
        if "cuda" in params.device:
            if torch.cuda.is_available():
                print(
                    f"Using CUDA device {torch.cuda.current_device()} with compute capability {torch.cuda.get_device_capability(0)}"
                )
                torch_helpers.device = torch.device(params.device)
            else:
                print("CUDA is not available")
        else:
            torch_helpers.device = torch.device(params.device)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    exit(main(params))
