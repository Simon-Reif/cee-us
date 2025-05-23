### for running FB learning and evaluation on offline playdata
import logging
import os
import pickle
import sys
import numpy as np
from smart_settings.param_classes import recursive_objectify
import torch
import wandb
from tqdm import tqdm
from collections import defaultdict

import yaml

from mbrl import allogger, torch_helpers
from mbrl.controllers.fb import ForwardBackwardController
from mbrl.environments import env_from_string
from mbrl.offline_helpers.buffer_manager import BufferManager
from mbrl.params_utils import read_params_from_cmdline, save_settings_to_json
from mbrl.seeding import Seeding
from mbrl.offline_helpers.checkpoints import get_latest_checkpoint, save_fb_checkpoint, save_meta
from mbrl.offline_helpers.eval import eval, print_best_success_by_task, update_best_success_by_task
from mbrl.workflow.name_runs_dirs import get_wandb_name, get_working_dir


def main(params):
    logger = allogger.get_logger(scope="main", basic_logging_params={"level": logging.INFO})
    Seeding.set_seed(params.seed if "seed" in params else None)
    # update params file with seed (either given or generated above)

    #TODO: refactor this later
    params_copy = params
    #params_copy = params._mutable_copy()
    params_copy["seed"] = Seeding.SEED
    
    #using params copy to use existing code defined on params with different tasks etc

    buffer_manager = BufferManager(params_copy)

    # buffer = load_buffer_wog(params_copy)
    # buffer_meta_path=os.path.join(params.training_data_dir, 'rollouts_meta.npy')
    # if os.path.exists(buffer_meta_path):
    #     stats_dict = np.load(buffer_meta_path, allow_pickle=True).item()
    #     obs_mean, obs_std = stats_dict["mean"], stats_dict["std"]
    # else:
    #     obs_mean, obs_std = buffer.get_mean_std()
    #     stats_dict={"mean": obs_mean, "std": obs_std}
    #     np.save(buffer_meta_path, stats_dict)


    obs_dim = buffer_manager.get_obs_dim()
    action_dim = buffer_manager.get_action_dim()
    params_copy.controller_params.model.obs_dim = obs_dim
    params_copy.controller_params.model.action_dim = action_dim

    debug=params.debug
    start_iter=0
    if "continue_training" in params_copy and params_copy.continue_training:
        fb_controller, idx = get_latest_checkpoint(params_copy.working_dir)
        print(f"Loaded agent from checkpoint {idx}")
        start_iter = idx
    elif "load_agent" in params_copy and params_copy.load_agent is not None:
        fb_controller = ForwardBackwardController.load(params.load_agent, params_copy.controller_params)
        print(f"Loaded agent from {params.load_agent}")
    else:
        fb_controller = ForwardBackwardController(params_copy.controller_params)

    #for normalization during interaction with env
    if params_copy.controller_params.model.norm_obs:
        obs_mean, obs_std = buffer_manager.get_mean_std()
        fb_controller.set_data_stats(obs_mean, obs_std)
    
    save_settings_to_json(params_copy, params.working_dir)

    # print("_______________Debugging_______________")
    # print(type(buffer))
    # print(buffer[0]["observations"].shape)
    # print("_______________Debugging_End_______________")

    best_success_by_task = defaultdict(dict)
    for iteration in tqdm(range(start_iter+1, start_iter+1+params.num_train_steps)):
    #for iteration in tqdm(range(10)):

        metrics = fb_controller.update(buffer_manager, iteration)

        if debug or (iteration % params.log_every_updates == 0):
            for k, v, in metrics.items():
                logger.log(torch_helpers.to_numpy(v), key=k)
            wandb.log(metrics, step=iteration)
            allogger.get_root().flush(children=True)


        ### Debug
        if debug:
            params_copy["number_of_rollouts"] = 2
        ## Debug End
        if debug or (iteration % params.eval.eval_every_steps == 0 and iteration > 0) or iteration == start_iter+params.num_train_steps:
            success_rates_eps, success_rates, z_r_dict, bs = eval(fb_controller, buffer_manager, params_copy, iteration)
            # TODO: move this into eval
            updated = update_best_success_by_task(best_success_by_task, success_rates, iteration, debug)
            if updated:
                print_best_success_by_task(best_success_by_task, to_yaml=True, working_dir=params.working_dir)

            save_fb_checkpoint(params.working_dir, iteration, success_rates_dict=success_rates_eps, 
                               controller=fb_controller, z_r_dict=z_r_dict, bs=bs, loud=2 if debug else 0)
            
        
    ###
    # End Main Loop
    ###
    save_meta(params_copy.working_dir, iteration)
    #TODO save final model
    #TODO maybe do sth with "total_metrics"
    print_best_success_by_task(best_success_by_task, console=True)
    allogger.close()

    return 0



if __name__ == "__main__":
    params = read_params_from_cmdline(verbose=True, save_params=False, make_immutable=False)
    wandb.login(key="25ee8d2e5fab3f028de5253bacadfe1ae8bfb760")
    
    #TODO: be careful whether this is automatically overwritten in sweep
    run = wandb.init(project=params.logging.project, entity="srtea", config=params)

    # run started as part of sweep without yaml config
    if len(sys.argv) < 2:
        params = recursive_objectify(run.config.as_dict(), make_immutable=False)
    
    if "set_dynamic_work_dir" in params and params.set_dynamic_work_dir:
        params.working_dir = get_working_dir(params, run)
    # if "set_dynamic_wandbname" in params and params.set_dynamic_wandbname:
    #     wandb.name = get_wandb_name(params, run)

    os.makedirs(params.working_dir, exist_ok=True)

    allogger.basic_configure(
        logdir=params.working_dir,
        default_outputs=["tensorboard"],
        manual_flush=True,
        tensorboard_writer_params=dict(min_time_diff_btw_disc_writes=1),
    )

    allogger.utils.report_env(to_stdout=True)

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
    run.finish()