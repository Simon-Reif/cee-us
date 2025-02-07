import logging
import os
import pickle

import numpy as np
import torch
import tqdm
from mbrl import allogger, torch_helpers
from mbrl.controllers.fb import ForwardBackwardController
from mbrl.environments import env_from_string
from mbrl.helpers import gen_rollouts
from mbrl.params_utils import read_params_from_cmdline, save_settings_to_json
from mbrl.rollout_utils import RolloutManager
from mbrl.rolloutbuffer import RolloutBuffer
from mbrl.seeding import Seeding


def calculate_success_rates(env, buffer: RolloutBuffer):
    success_rate = []
    for i in range(len(buffer)):
        rollout_success = env.eval_success(buffer[i]["next_observations"])
        stable_T = 5
        if env.name == "FetchPickAndPlaceConstruction" and "tower" in env.case:
            # Stack is only successful if we have a full tower! 
            # Check if the tower is stable for at least 5 timesteps
            dy = np.diff(rollout_success)
            success = np.logical_and(rollout_success[1:]==env.num_blocks, dy==0)
            success_rate.append(np.sum(success)>stable_T)
        elif env.name == "FetchPickAndPlaceConstruction" and env.case == 'PickAndPlace':
            # We determine success as highest number of solved elements with at least 5 timesteps of success
            u, c = np.unique(rollout_success, return_counts = True)
            # u: unique values, c: counts
            count_of_highest_success = c[np.argmax(u)]    
            success_rate.append(u[c>stable_T][-1]/env.nObj)
        else:
            # For flip, throw and Playground env push tasks: just get success at the end of rollout
            success_rate.append(rollout_success[-1]/env.nObj)
    print("Success rate over {} rollouts in task {}, is {}".format(len(buffer), env.case, np.asarray(success_rate).mean()))#
    return success_rate

#self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
#self.logger.log(torch.min(self.costs_).item(), key="best_trajectory_cost")

def eval(controller: ForwardBackwardController, offline_data: RolloutBuffer, params, t=None):
    eval_logger = allogger.get_logger(scope="eval", default_outputs=["tensorboard"])
    checkpoint_dir = os.path.join(params.working_dir, f"checkpoint_{t}")
    """"
    num_eval_episodes: 10
    num_inference_samples: 50_000
    eval_tasks: None
    """
    # TODO: maybe inference_batch_size parameter necessary
    if t is not None:
        print(f"Evaluation at iteration {t}")
    obs, actions, next_obs = torch_helpers.to_tensor_device(offline_data.get_random_transitions(params.num_inference_samples))
    bs=controller.calculate_Bs(next_obs)
    for task in params.eval_tasks:
        subdir = os.path.join(params.working_dir, "eval", task)
        params["env_params"]["case"] = task
        env = env_from_string(params.env, **params.env_params)
        z_r = controller.estimate_z_r(obs, actions, next_obs, env, bs=bs)
        controller.set_zr(z_r)
        rollout_man = RolloutManager(env, params.rollout_params)
        rollout_buffer = RolloutBuffer()
        #TODO: set params.num_rollouts to eval.num_eval_episodes
        rollout_buffer = gen_rollouts(
            params,
            rollout_man,
            controller,
            None, #initial_controller
            rollout_buffer,
            None, #forward_model
            t, #iteration
            False, #do_initial_rollouts
        )
        success_rates= calculate_success_rates(env, rollout_buffer)
        #TODO: also record rewards
        #TODO: save success rates, zr, 



#TODO: option to continue training from a checkpoint
def main(params):
    logger = allogger.get_logger(scope="main", basic_logging_params={"level": logging.INFO})
    Seeding.set_seed(params.seed if "seed" in params else None)
    # update params file with seed (either given or generated above)
    params_copy = params._mutable_copy()
    params_copy["seed"] = Seeding.SEED
    save_settings_to_json(params_copy, params.working_dir)
    
    #using params copy to use existing code defined on params with different tasks etc
    
    with open(os.path.join(params.training_data_dir, 'rollouts'), 'rb') as f:
        buffer = pickle.load(f)
    obs_dim=buffer[0]["observations"].shape[-1]
    act_dim=buffer[0]["actions"].shape[-1]
    fb_controller = ForwardBackwardController(params.controller_params, obs_dim=obs_dim, act_dim=act_dim)
    # print("_______________Debugging_______________")
    # print(type(buffer))
    # print(buffer[0]["observations"].shape)
    # print("_______________Debugging_End_______________")

    for iteration in tqdm(range(params.num_train_steps)):
        
        
        if iteration % params.eval_every == 0 and iteration > 0:
            eval(fb_controller, buffer, params_copy, iteration)
        


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
