import yaml
import numpy as np
from copy import deepcopy
from mbrl import torch_helpers
from mbrl.controllers.fb import ForwardBackwardController
from mbrl.environments import env_from_string
from mbrl.helpers import gen_rollouts
from mbrl.rollout_utils import RolloutManager
from mbrl.rolloutbuffer import RolloutBuffer

# returns success rates for all rollouts, so we can compare to individual rollouts, calcuate std, etc
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
    #print("Success rate over {} rollouts in task {}, is {}".format(len(buffer), env.case, np.asarray(success_rate).mean()))#
    return np.array(success_rate)

#self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
#self.logger.log(torch.min(self.costs_).item(), key="best_trajectory_cost")

def eval(controller: ForwardBackwardController, offline_data: RolloutBuffer, params, t=None, debug=False):
    #eval_logger = allogger.get_logger(scope="eval", default_outputs=["tensorboard"])
    """"
    num_eval_episodes: 10
    num_inference_samples: 50_000
    eval_tasks: None
    """
    # TODO: maybe inference_batch_size parameter necessary
    if t is not None:
        print(f"Evaluation at iteration {t}")
    obs, actions, next_obs = torch_helpers.to_tensor_device(*offline_data.get_random_transitions(params.eval.num_inference_samples))
    bs=controller.calculate_Bs(next_obs)
    results={}
    z_rs={}
    for task in params.eval.eval_tasks:
        #TODO: load entire task settings, and task horizon from rollout params
        task_params=params.eval_envs[task]
        env=task_params.env
        env_params=task_params.env_params
        env = env_from_string(env, **env_params)
        params.rollout_params.task_horizon = task_params.rollout_params.task_horizon
        print(f"Evaluating on task {task} with horizon {params.rollout_params.task_horizon}")
        rollout_man = RolloutManager(env, params.rollout_params)
        z_r = controller.estimate_z_r(obs, actions, next_obs, env, bs=bs)
        controller.set_zr(z_r)
        if debug:
            print(f"Calculated zr: {z_r}")
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
        print("Success rate over {} rollouts in task {}, is {}".format(len(rollout_buffer), task, success_rates.mean()))#

        results[task] = success_rates
        z_rs[task] = torch_helpers.to_numpy(z_r)
    return results, z_rs, bs


def update_best_success_by_task(best_success_by_task, success_rates_dict, iteration, debug=False):
    for task, success_rates in success_rates_dict.items():
        success_rate = success_rates.mean()
        updated = False
        if task not in best_success_by_task or success_rate > best_success_by_task[task]["success_rate"]:
            best_success_by_task[task]["iter"] = iteration
            best_success_by_task[task]["success_rate"] = success_rate
            updated = True
            if debug:
                print(f"New best success rate for task {task} is {success_rate} at iteration {iteration}")
    return updated

def _to_simple_types(numpy_dict):
    for key, value in numpy_dict.items():
        if isinstance(value, dict):
            _to_simple_types(value)
        else:
            try:
                numpy_dict[key] = value.item()
            except AttributeError:
                pass

def print_best_success_by_task(best_success_by_task, console=False, to_yaml=False, working_dir=None):
    if to_yaml:
        print_dict = deepcopy(best_success_by_task)
        _to_simple_types(print_dict)
        with open(f"{working_dir}/best_success_rates.yaml", "w") as f:
            yaml.dump(print_dict, f, default_flow_style=False, sort_keys=False)
    if console:
        for task, success in best_success_by_task.items():
            print(f"Best success rate for task {task} is {success['success_rate']} at iteration {success['iter']}")
    