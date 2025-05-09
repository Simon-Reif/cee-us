import yaml
import wandb
import time
import numpy as np
from copy import deepcopy
from mbrl import allogger, torch_helpers
from mbrl.controllers.fb import ForwardBackwardController
from mbrl.environments import env_from_string
from mbrl.helpers import gen_rollouts
from mbrl.offline_helpers.buffer_manager import BufferManager
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
            # Check if the tower is stable for at least 5 timesteps ##comment: at least 7 time steps actually
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

def eval(controller: ForwardBackwardController, offline_data: BufferManager, params, t=None, debug=False):
    eval_logger = allogger.get_logger(scope="eval", default_outputs=["tensorboard"])
    """"
    num_eval_episodes: 10
    num_inference_samples: 50_000
    eval_tasks: None
    """
    # TODO: maybe inference_batch_size parameter necessary
    if t is not None:
        print(f"Evaluation at iteration {t}")

    inference_samples = offline_data.sample(params.eval.num_inference_samples)
    next_obs = torch_helpers.to_tensor(inference_samples["next_observations"]).to(torch_helpers.device)
    #obs, actions, next_obs = torch_helpers.to_tensor_device(*offline_data.get_random_transitions(params.eval.num_inference_samples))

    bs=controller.calculate_Bs(next_obs)
    success_rates_eps_dict={}
    success_rates_dict={}
    z_rs={}
    for task in params.eval.eval_tasks:
        task_params=params.eval_envs[task]
        ### Task Adaptation
        env=task_params.env
        env_params=task_params.env_params
        env = env_from_string(env, **env_params)
        params.rollout_params.task_horizon = task_params.rollout_params.task_horizon
        print(f"Evaluating on task {task} with horizon {params.rollout_params.task_horizon}")
        params.rollout_params.obs_wo_goals = True
        rollout_man = RolloutManager(env, params.rollout_params)
        #TODO: set goals for obs from buffer
        goals = [env._sample_goal().copy() for _ in range(params.eval.num_inference_goals)]
        goals = np.array(goals).repeat(len(next_obs)/len(goals), axis=0)
        #this only uses observations in calculating rewards since bs fixed between tasks
        z_r = controller.estimate_z_r(next_obs, goals, env, bs=bs)
        controller.set_zr(z_r)
        if debug:
            print(f"Calculated zr: {z_r}")
        ### Task Adaptation End
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
        success_rates_eps = calculate_success_rates(env, rollout_buffer)
        mean_success_rate = success_rates_eps.mean()
        eval_logger.log(mean_success_rate, key=f"{task}_success_rate")
        wandb.log({f"eval/{task}_success_rate": mean_success_rate}, step=t)
        print("Success rate over {} rollouts in task {}, is {}".format(len(rollout_buffer), task, mean_success_rate))#

        success_rates_eps_dict[task] = success_rates_eps
        success_rates_dict[task] = mean_success_rate
        z_rs[task] = torch_helpers.to_numpy(z_r)
    bs = torch_helpers.to_numpy(bs)
    return success_rates_eps_dict, success_rates_dict, z_rs, bs


def update_best_success_by_task(best_success_by_task, success_rates, iteration, debug=False):
    updated = False
    for task, success_rate in success_rates.items():
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
    

def _find_match(obs, waypoint, threshold):
    found =False
    i=0
    smallest_dist=None
    while not found and i<len(obs):
        dist=np.linalg.norm(obs[i]-waypoint)
        if dist<threshold:
            found=True
            idx=i
        if smallest_dist is None:
            smallest_dist=dist
        elif smallest_dist>dist:
            smallest_dist=dist
            idx=i
        i+=1
    if found:
        return idx, dist, found
    else:
        return idx, smallest_dist, found
    

# return indices of matched waypoints, distances, under_threshold, and if all waypoints were matched
def _discr_match(obs, ctrl_waypoints, threshold):
    idx, dist, under_thresh  = _find_match(obs, ctrl_waypoints[0], threshold)
    obs_rest=obs[idx:]
    ctrl_waypoints_rest=ctrl_waypoints[1:]
    n_obs = len(obs_rest)
    n_wp = len(ctrl_waypoints_rest)
    if n_obs<n_wp:
        return [idx], [dist], [under_thresh], False
    if n_wp==0:
        return [idx], [dist], [under_thresh], True
    idcs, dists, under_thresholds, all_matched = _discr_match(obs_rest, ctrl_waypoints_rest, threshold)
    idcs = [i+idx for i in idcs]
    idcs.insert(0, idx)
    dists.insert(0, dist)
    under_thresholds.insert(0, under_thresh)
    return idcs, dists, under_thresholds, all_matched



def on_distr_motion_tracking(controller: ForwardBackwardController, buffer:RolloutBuffer, env, params, t=None, debug=False):
    num_eval_traj = 100
    num_states_to_match=10
    #TODO: set threshold to what makes sense from observation space
    threshold=0.5
    traj_ids = np.random.choice(len(buffer), num_eval_traj)
    state_ids = np.linspace(0, len(buffer[0])-1, num_states_to_match, dtype=int)
    params.num_rollouts = 1
    for traj_id in traj_ids:
        traj = buffer[traj_id]
        obs = traj["next_observations"]
        z_r = controller.zr_from_states(obs[state_ids])
        controller.set_zr(z_r)
        ctrl_rollouts = RolloutBuffer()
        rollout_man = RolloutManager(env, params.rollout_params, start_state=obs[0])
        ctrl_rollouts = gen_rollouts(
            params,
            rollout_man,
            controller,
            None, #initial_controller
            ctrl_rollouts,
            None, #forward_model
            t, #iteration
            False, #do_initial_rollouts
        )
        ctrl_waypoints=ctrl_rollouts[0]["next_observations"][state_ids]
        idcs, dists, under_thresh, all_matched = _discr_match(obs, ctrl_waypoints)


if __name__ == "__main__":
    a = np.array([10, 4, 9, 3, 11, 1, 1, 7, 3], dtype=np.float32)
    b = np.array([10.9, 1, 1.1, 7.2], dtype=np.float32)
    print(a.dtype, b.dtype)
    start_t = time.time()
    print(_discr_match(a, b, 0.5))
    print(f"Elapsed time: {time.time()-start_t}")