### for running FB learning and evaluation on offline playdata
import logging
import os
import pickle
import numpy as np
import torch
import wandb
from tqdm import tqdm
from collections import defaultdict

import yaml

from mbrl import allogger, torch_helpers
from mbrl.controllers.bc import BehaviorCloningController
from mbrl.environments import env_from_string
from mbrl.helpers import gen_rollouts
from mbrl.offline_helpers.buffer_manager import BufferManager
from mbrl.offline_helpers.buffer_utils import load_buffer_wog
from mbrl.params_utils import read_params_from_cmdline, save_settings_to_json
from mbrl.rollout_utils import RolloutManager
from mbrl.rolloutbuffer import RolloutBuffer
from mbrl.seeding import Seeding
from mbrl.offline_helpers.checkpoints import get_latest_checkpoint, save_fb_checkpoint, save_meta
from mbrl.offline_helpers.eval import calculate_success_rates, print_best_success_by_task, update_best_success_by_task
from mbrl.workflow.name_runs_dirs import get_working_dir


def eval_bc(controller, params, t, train_data:BufferManager=None):
    if t is not None:
        print(f"Evaluation at iteration {t}")
    success_rates_eps_dict={}
    success_rates_dict={}
    for task in params.eval.eval_tasks:
        task_params=params.eval_envs[task]
        env=task_params.env
        env_params=task_params.env_params
        env = env_from_string(env, **env_params)
        params.rollout_params.task_horizon = task_params.rollout_params.task_horizon
        print(f"Evaluating on task {task} with horizon {params.rollout_params.task_horizon}")
        #params.rollout_params.obs_wo_goals = True
        rollout_man = RolloutManager(env, params.rollout_params)
        rollout_buffer = RolloutBuffer()
        if train_data is not None:
            start_states = train_data.sample_start_states(params.number_of_rollouts)
        else:
            start_states = None
        # setting start_states also sets goals!
        rollout_buffer = gen_rollouts(
            params,
            rollout_man,
            controller,
            None, #initial_controller
            rollout_buffer,
            None, #forward_model
            t, #iteration
            False, #do_initial_rollouts
            start_states=start_states,
        )
        success_rates_eps = calculate_success_rates(env, rollout_buffer)
        mean_success_rate = success_rates_eps.mean()
        wandb.log({f"eval/{task}_success_rate": mean_success_rate}, step=t)
        print("Success rate over {} rollouts in task {}, is {}".format(len(rollout_buffer), task, mean_success_rate))#

        success_rates_eps_dict[task] = success_rates_eps
        success_rates_dict[task] = mean_success_rate
    return success_rates_eps_dict, success_rates_dict

    

#TODO: option to continue training from a checkpoint
def main(params):
    logger = allogger.get_logger(scope="main", basic_logging_params={"level": logging.INFO})
    Seeding.set_seed(params.seed if "seed" in params else None)
    # update params file with seed (either given or generated above)
    params_copy = params._mutable_copy()
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


    # obs_dim=buffer[0]["observations"].shape[-1]
    # action_dim=buffer[0]["actions"].shape[-1]

    # TODO: instead define on observations with goals
    obs_dim=buffer_manager.get_obs_dim()
    action_dim=buffer_manager.get_action_dim()
    params_copy.controller_params.model.obs_dim = obs_dim
    params_copy.controller_params.model.action_dim = action_dim

    debug=params.debug
    if debug:
        print(f"Obs dim: {obs_dim}, Action dim {action_dim}")
    start_iter=0
    
    bc_controller = BehaviorCloningController(params_copy.controller_params)

    if params_copy.controller_params.model.norm_obs:
        obs_mean, obs_std = buffer_manager.get_mean_std()
        bc_controller.set_data_stats(obs_mean, obs_std)

    save_settings_to_json(params_copy, params.working_dir)

    # print("_______________Debugging_______________")
    # print(type(buffer))
    # print(buffer[0]["observations"].shape)
    # print("_______________Debugging_End_______________")

    best_success_by_task = defaultdict(dict)
    for iteration in tqdm(range(start_iter+1, start_iter+1+params.num_train_steps)):
    #for iteration in tqdm(range(10)):

        metrics = bc_controller.update(buffer_manager, iteration)

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
            if params_copy.eval.start_states_from_train_data:
                success_rates_eps, success_rates = eval_bc(bc_controller, params_copy, iteration, train_data=buffer_manager)
            else:
                success_rates_eps, success_rates = eval_bc(bc_controller, params_copy, iteration)
            # TODO: move this into eval
            updated = update_best_success_by_task(best_success_by_task, success_rates, iteration, debug)
            if updated:
                print_best_success_by_task(best_success_by_task, to_yaml=True, working_dir=params.working_dir)

            save_fb_checkpoint(params.working_dir, iteration, success_rates_dict=success_rates_eps, 
                               controller=bc_controller, loud=2 if debug else 0)
            
        
        

    ###
    # End Main Loop
    ###
    save_meta(params_copy.working_dir, iteration)
    #TODO save final model
    #TODO generate summary (plot) of best results
    #TODO maybe do sth with "total_metrics"
    print_best_success_by_task(best_success_by_task, console=True)
    allogger.close()

    return 0



if __name__ == "__main__":
    params = read_params_from_cmdline(verbose=True, save_params=False, make_immutable=False)

    os.makedirs(params.working_dir, exist_ok=True)

    wandb.login(key="25ee8d2e5fab3f028de5253bacadfe1ae8bfb760")
    run=wandb.init(project=params.logging.project, entity="srtea", group="bc", config=params, allow_val_change=True)

    if "set_dynamic_work_dir" in params and params.set_dynamic_work_dir:
        #TODO: also set working_dir in wandb config
        params.working_dir = get_working_dir(params, run)
        run.config.update({"working_dir": params.working_dir}, allow_val_change=True)

    
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