import copy
import yaml
import wandb
import time
import numpy as np
from copy import deepcopy
from mbrl import allogger, torch_helpers
from mbrl.controllers.fb import ForwardBackwardController
from mbrl.environments import env_from_string
from mbrl.environments.fpp_construction_env import FetchPickAndPlaceConstruction
from mbrl.helpers import gen_rollouts
from mbrl.offline_helpers.buffer_manager import BufferManager
from mbrl.rollout_utils import RolloutManager
from mbrl.rolloutbuffer import RolloutBuffer


#for eval/adapt on data not in rolloutbuffer (and reference if other data trained on)
exp_data_dirs = {
    "planner_flip": "results/cee_us/zero_shot/2blocks/225iters/flip_4500/gnn_ensemble_icem/checkpoints_000/filtered",
    "planner_throw": "results/cee_us/zero_shot/2blocks/225iters/throw_4500/gnn_ensemble_icem/checkpoints_000/filtered",
    "planner_pp": "results/cee_us/zero_shot/2blocks/225iters/pp_4500/gnn_ensemble_icem/checkpoints_000/filtered",
    "planner_stack": "results/cee_us/zero_shot/2blocks/225iters/stack_4500/gnn_ensemble_icem/checkpoints_000/filtered",
}

# returns success rates for all rollouts, so we can compare to individual rollouts, calcuate std, etc
def calculate_success_rates(env: FetchPickAndPlaceConstruction, buffer: RolloutBuffer):
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
        elif env.name == "FetchPickAndPlaceConstruction" and env.case == 'Reach':
            # maybe replace this with success anywhere or for a while?
            success_rate.append(rollout_success[-1])
        else:
            # For flip, throw and Playground env push tasks: just get success at the end of rollout
            success_rate.append(rollout_success[-1]/env.nObj)
    #print("Success rate over {} rollouts in task {}, is {}".format(len(buffer), env.case, np.asarray(success_rate).mean()))#
    return np.array(success_rate)

def maybe_set_start_states(buffer_manager, params, task):
    if "where_appl_start_states_expert" in params and params.eval.where_appl_start_states_expert:
        exp_buffer = buffer_manager.maybe_get_expert_buffer(task)
        if exp_buffer:
            if params.debug:
                print(f"Using expert buffer {exp_buffer.name} from training data for start states")
            return exp_buffer.sample_start_states(params.number_of_rollouts)
    if params.eval.start_states_from_train_data:
        if params.debug:
            print("Using training data for start states")
        return buffer_manager.sample_start_states(params.number_of_rollouts)
    else:
        if params.debug:
            print("Using random start states")
        return None
    
def eval(controller: ForwardBackwardController, offline_data: BufferManager, params, t=None, final=False, debug=False):
    eval_logger = allogger.get_logger(scope="eval", default_outputs=["tensorboard"])
    """"
    num_eval_episodes: 10
    num_inference_samples: 50_000
    eval_tasks: None
    """
    # TODO: maybe inference_batch_size parameter necessary
    if t is not None:
        print(f"Evaluation at iteration {t}")
    if final:
        params.number_of_rollouts = params.eval.number_of_rollouts_final


    inference_samples = offline_data.sample(params.eval.num_inference_samples)
    next_obs = torch_helpers.to_tensor(inference_samples["next_observations"]).to(torch_helpers.device)
    #obs, actions, next_obs = torch_helpers.to_tensor_device(*offline_data.get_random_transitions(params.eval.num_inference_samples))

    bs=controller.calculate_Bs(next_obs)
    success_rates_eps_dict={}
    success_rates_dict={}
    z_rs={}
    goals={}
    for task in params.eval.eval_tasks:
        task_params=params.eval_envs[task]
        ### Task Adaptation
        env=task_params.env
        env_params=task_params.env_params
        env = env_from_string(env, **env_params)
        params.rollout_params.task_horizon = task_params.rollout_params.task_horizon
        print(f"Evaluating on task {task} with horizon {params.rollout_params.task_horizon}")
        rollout_man = RolloutManager(env, params.rollout_params)
        #TODO: set goals for obs from buffer
        #TODO: set just one goal, use that goal in the env
        start_states = maybe_set_start_states(offline_data, params, task)
        if params.eval.where_appl_start_states_expert or params.eval.start_states_from_train_data:
            goal = env.goal_from_state(start_states[0])
        else:
            goal = env._sample_goal()
        env.set_fixed_goal(goal)
        #this only uses observations in calculating rewards since bs fixed between tasks
        z_r = controller.estimate_z_r(next_obs, goal, env, bs=bs)
        controller.set_zr(z_r)
        # if debug:
        #     print(f"Calculated zr: {z_r}")
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
            start_states=start_states,
        )
        success_rates_eps = calculate_success_rates(env, rollout_buffer)
        mean_success_rate = success_rates_eps.mean()
        eval_logger.log(mean_success_rate, key=f"{task}_success_rate")
        wandb.log({f"eval/{task}_success_rate": mean_success_rate}, step=t)
        print("Success rate over {} rollouts in task {}, is {}".format(len(rollout_buffer), task, mean_success_rate))#

        success_rates_eps_dict[task] = success_rates_eps
        success_rates_dict[task] = mean_success_rate
        z_rs[task] = torch_helpers.to_numpy(z_r)
        goals[task] = goal

    eval_return_dict = {
        "success_rates_eps": success_rates_eps_dict,
        "success_rates": success_rates_dict,
        "z_rs": z_rs,
        "goals": goals,
    }
    return eval_return_dict


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
    

def _random_uniform_indices(length_arr, ratio):
    num_indices = np.round(length_arr*ratio).astype(int)
    return np.random.choice(length_arr, num_indices, replace=False)


#TODO: pretty ugly, maybe refactor
def s_eval(controller: ForwardBackwardController, buffer_manager: BufferManager, params, t=None, debug=False):

    #adaptation_modalities: ["gr", "im10", "rew10", "rew50"]
    adapt_modalities = params.eval.s_eval.adaptation_modalities

    #to set number_of_rollouts and rollout_params.task_horizon
    eval_rollout_params = copy.deepcopy(params.rollout_params)

    num_eval_traj = params.eval.s_eval.num_eval_trajectories # or change name
    task_traj_tuples = buffer_manager.get_tasks_and_traj_from_exp_data(num_eval_traj)
    for task, exp_trajectories in task_traj_tuples:
        rollout_buffers = {adapt_mod: RolloutBuffer() for adapt_mod in adapt_modalities}
        task_params=params.eval_envs[task]
        ### Task Adaptation
        env=task_params.env
        env_params=task_params.env_params
        env = env_from_string(env, **env_params)
        rollout_man = RolloutManager(env, eval_rollout_params)
        
        for exp_traj in exp_trajectories:
            len_traj = len(exp_traj["observations"])
            rollout_man.task_horizon = len_traj
            start_state = exp_traj["env_states"][0]
            env.set_GT_state(start_state)
            goal_positions = env.goal
            assert all(goal_positions == env.goal_from_state(start_state)), "Goal positions from state and env goal do not match"

            #task adaptation

            next_obs_torch = torch_helpers.to_tensor(exp_traj["next_observations"]).to(torch_helpers.device)
            rewards_exp = exp_traj["rewards"]

            # Task Adaptation and Rollout Generation
            if "gr" in adapt_modalities:
                #maybe dimensionality a problem-> test
                z_r = controller.zr_from_obs(next_obs_torch[-1])
                controller.set_zr(z_r)
                rollout = rollout_man.single_rollout(controller, start_state)
                rollout_buffers["gr"].extend(rollout)
            idcs10 = None
            if "im10" in adapt_modalities:
                idcs10 = _random_uniform_indices(len_traj, 0.1)
                # maybe need to use a different function here rather than just indexing
                Bs = controller.calculate_Bs(next_obs_torch[idcs10])
                z_r = controller.zr_from_obs(bs=Bs)
                controller.set_zr(z_r)
                rollout = rollout_man.single_rollout(controller, start_state)
                rollout_buffers["im10"].extend(rollout)
            if "rew10" in adapt_modalities:
                if idcs10 is None:
                    idcs10 = _random_uniform_indices(len_traj, 0.1)
                    Bs = controller.calculate_Bs(next_obs_torch[idcs10])
                rew = rewards_exp[idcs10]
                z_r = controller.zr_from_obs_and_rews(bs=Bs, rewards=rew)
                controller.set_zr(z_r)
                rollout = rollout_man.single_rollout(controller, start_state)
                rollout_buffers["rew10"].extend(rollout)
            if "rew50" in adapt_modalities:
                idcs50 = _random_uniform_indices(len_traj, 0.5)
                Bs = controller.calculate_Bs(next_obs_torch[idcs50])
                rew = rewards_exp[idcs50]
                z_r = controller.zr_from_obs_and_rews(bs=Bs, rewards=rew)
                controller.set_zr(z_r)
                rollout = rollout_man.single_rollout(controller, start_state)
                rollout_buffers["rew50"].extend(rollout)

        # Evaluation
        # disc = params.controller_params.train.discount
        # avg_rew_exp = np.array([np.mean(exp_traj.avg_disc_reward(disc)) for exp_traj in exp_trajectories])
        avg_rew_exp = np.array([np.mean(exp_traj.avg_reward()) for exp_traj in exp_trajectories])

        for adapt_modality in adapt_modalities:
            metric_prefix = f"s_eval/{task}/{adapt_modality}/"
            buffer = rollout_buffers[adapt_modality]
            
            #rew_fb = buffer.get_mean_disc_rewards_per_rollout(disc)
            rew_fb = buffer.get_mean_rewards_per_rollout()

            mean_rel_rew = np.mean(rew_fb / avg_rew_exp)
            wandb.log({f"{metric_prefix}mean_relative_reward_undisc": mean_rel_rew}, step=t)

            success_rates_eps = calculate_success_rates(env, buffer)
            wandb.log({f"{metric_prefix}success_rate": np.mean(success_rates_eps)}, step=t)

            #TODO: make and return nested dict of metrics
