

#TODO: loading agents, zr, goals
#TODO: collecting rollouts (with and without set zr)
#TODO: record video

import copy
import os

import imageio
import numpy as np
import torch
from mbrl import allogger, torch_helpers
from mbrl.environments import env_from_string
from mbrl.offline_helpers.buffer_manager import BufferManager
from mbrl.offline_helpers.buffer_utils import get_buffer_wo_goals
from mbrl.offline_helpers.checkpoints import _get_cp_dir, load_checkpoint
from mbrl.offline_helpers.eval import _random_uniform_indices, calculate_success_rates, eval_task, next_obs_and_Bs_from_buffer
from mbrl.offline_helpers.util_funs import dynamic_time_warp
from mbrl.rollout_utils import RolloutManager
from mbrl.rolloutbuffer import RolloutBuffer


def set_rollout_params(params, num_rollouts, task_horizon):
    new_params = params.copy()
    new_params.number_of_rollouts = num_rollouts
    new_params.rollout_params.task_horizon = task_horizon
    return new_params

#from cee-us freeplay
default_env= {
    "env": "FetchPickAndPlaceConstruction",
    "env_params": {
        "case": "Singletower",
        "num_blocks": 2,
        "shaped_reward": True,
        "sparse": False,
        "stack_only": True,
        "visualize_mocap": False,
        "visualize_target": False
        }
    }

#TODO: make all task env available globally
#if task doesn't play a role
def get_default_env():
    env_name = default_env["env"]
    env_params = default_env["env_params"]
    env = env_from_string(env_name, **env_params)
    return env

#maybe globally save location of training data



#TODO: maybe rename
class Replay_Manager:
    def __init__(self, working_dir=None, cp_dir=None, iter=None):
        self.cp_dir = cp_dir if cp_dir is not None else _get_cp_dir(working_dir, iter)
        self.working_dir = working_dir if working_dir is not None else os.path.dirname(self.cp_dir)
        cp_dict = load_checkpoint(cp_dir=self.cp_dir)
        self.params = cp_dict["params"]
        self.fb_controller = cp_dict["fb_controller"]
        self.zr_dict = cp_dict.get("zr_dict", None)
        self.goal_dict = cp_dict.get("goal_dict", None)
        self.buffer_manager = None  # will be set in load_training_data
        self.render = False
        allogger.basic_configure(
            logdir=os.path.join(self.working_dir, "replay_man_logs"),
            default_outputs=["tensorboard"],
            manual_flush=True,
            tensorboard_writer_params=dict(min_time_diff_btw_disc_writes=1),
        )
        self.imit_cost_fun = lambda x, y: np.linalg.norm(x - y, axis=-1)  # default cost for imitation

    def set_wandb_run(self, wandb_run):
        self.wandb_run = wandb_run
    def load_training_data(self, params=None, reload=False):
        if self.buffer_manager is not None and not reload:
            return self.buffer_manager
        params = params if params is not None else self.params
        self.buffer_manager = BufferManager(params)
        return self.buffer_manager
    
    def get_env(self, task):
        task_params=self.params.eval.eval_envs[task]
        env=task_params.env
        env_params=task_params.env_params
        env = env_from_string(env, **env_params)
        return env

    #TODO: select traj, record traj and imitation/goal reaching

    def imitation(self, episode, env=None, adapt_ratio=1.0, num_traj=1, extra_steps=0, cost_fn=None):
        if env is None:
            env = get_default_env()
        idcs = _random_uniform_indices(len(episode), adapt_ratio)
        next_obs = torch_helpers.to_tensor(episode["next_observations"]).to(torch_helpers.device)
        z_r = self.fb_controller.zr_from_obs(next_obs[idcs])
        self.fb_controller.set_zr(z_r)
        roll_params = copy.deepcopy(self.params.rollout_params)
        roll_params.task_horizon = len(episode) + extra_steps
        rollout_man = RolloutManager(env, roll_params)
        start_state = episode["env_states"][0]
        rollouts = rollout_man.sample(self.fb_controller, render=self.render, no_rollouts=num_traj
                                      ,start_state=[start_state]*num_traj)  
        rollouts = get_buffer_wo_goals(RolloutBuffer(rollouts=rollouts), env)
        if cost_fn is None:
            cost_fn = self.imit_cost_fun 
        dtw_dists = [dynamic_time_warp(episode["observations"], rollout["observations"], cost_fn) for rollout in rollouts]
        print(f"DTW distances: {dtw_dists}")
        info_dict = {"dtw_dists": dtw_dists, "z_r": z_r}
        return rollouts, info_dict

    #deprecated, maybe rewrite
    #default to num_rollouts and task horizons in params
    def gen_rollouts_task(self, task, num_rollouts, task_horizon=None, use_saved_goal_zr=True, Bs=None, next_obs=None, eval=True):
        task_params=self.params.eval_envs[task]
        env=task_params.env
        env_params=task_params.env_params
        env = env_from_string(env, **env_params)

        roll_params = copy.deepcopy(self.params.rollout_params)
        roll_params.task_horizon = task_horizon if task_horizon is not None else task_params.rollout_params.task_horizon
        print(f"Generating {num_rollouts} rollouts on task {task} with horizon {roll_params.task_horizon}")
        #NOTE: maybe this needs allogger
        rollout_man = RolloutManager(env, roll_params)
        print(f"use_saved_goal_zr: {use_saved_goal_zr}")
        if use_saved_goal_zr:
            if not task in self.zr_dict or not task in self.goal_dict:
                raise ValueError(f"If use_goal_zrs is True, zr_dict and goal_dict must contain the task {task}.")
            goal = self.goal_dict[task]
            zr = self.zr_dict[task]
        else:
            goal = env._sample_goal()
            if next_obs is None:
                samples = self.buffer_manager.sample(self.params.eval.num_inference_samples)
                next_obs = torch_helpers.to_tensor(samples["next_observations"]).to(torch_helpers.device)
            zr = self.fb_controller.estimate_z_r(next_obs, goal, env, bs=Bs)
        self.fb_controller.set_zr(zr)
        #TODO: maybe enable setting start states
        rollouts = rollout_man.sample(self.fb_controller, render=self.render, no_rollouts=num_rollouts)
        if eval:
            success_rates_eps = calculate_success_rates(env, rollouts)
            mean_success_rate = success_rates_eps.mean()
            print(f"Mean success rate for task {task}: {mean_success_rate}")
            return rollouts, {
                "success_rates_eps": success_rates_eps,
                "mean_success_rate": mean_success_rate,
                "zr": zr,
                "goal": goal,
            }
        else:
            return rollouts, {}
        
    #deprecated, maybe rewrite
    def generate_rollouts(
        self,
        tasks=None,
        num_rollouts=None,   
        task_horizon=None,  
        num_rollouts_list=None,  # alternative to num_rollouts, list of len(tasks)
        task_horizons=None,  # alternative to task_horizon, list of len(tasks)
        use_saved_goals_zrs=False, 
    ):
        if use_saved_goals_zrs:
            if self.zr_dict is None or self.goal_dict is None:
                raise ValueError("If use_goals_zrs is True, zr_dict and goal_dict must be provided.")
            next_obs=None
            Bs=None
        else:
            self.load_training_data()
            #same Bs for all tasks
            next_obs, Bs = next_obs_and_Bs_from_buffer(self.buffer_manager, self.fb_controller, self.params.eval.num_inference_samples)


        rollouts_dict = {}
        results_dicts = {}
        if tasks is None:
            tasks = self.params.eval.eval_tasks
        if num_rollouts is not None:
            num_rollouts_list = [num_rollouts] * len(tasks)
        elif num_rollouts_list is None:
            num_rollouts_list = [self.params.num_rollouts] * len(tasks)
        if task_horizon is not None:
            task_horizons = [task_horizon] * len(tasks)
        elif task_horizons is None:
            task_horizons = [None] * len(tasks)
        
        for i, task in enumerate(tasks):
            num_rollouts = num_rollouts_list[i]
            task_horizon = task_horizons[i]
        
            rollout, results_dict = self.gen_rollouts_task(
                task=task,
                num_rollouts=num_rollouts,
                task_horizon=task_horizon,
                next_obs=next_obs,
                Bs=Bs,
                use_saved_goal_zr=use_saved_goals_zrs,
            )  
            rollouts_dict[task] = rollout
            results_dicts[task] = results_dict
        return rollouts_dict, results_dicts
    
    #TODO: add options to set task horizons, etc
    def eval(self, tasks=None, num_rollouts=None, task_horizons=None, log_summary=False):
        """
        Evaluate the agent on the specified tasks.
        If tasks is None, use the tasks from the params.
        """
        self.load_training_data()
        if tasks is None:
            tasks = self.params.eval.eval_tasks
        if task_horizons is None:
            task_horizons = [None] * len(tasks)
        
        params_flex = copy.deepcopy(self.params)

        next_obs, bs = next_obs_and_Bs_from_buffer(self.buffer_manager, self.fb_controller, self.params.eval.num_inference_samples)

        rollouts_dict = {}
        success_rates_eps_dict={}
        success_rates_dict={}
        z_rs={}
        goals={}
        for i, task in enumerate(tasks):
            task_return_dict = eval_task(
                task=task,
                controller=self.fb_controller,
                params=params_flex,
                buffer_manager=self.buffer_manager,
                number_of_rollouts=num_rollouts,
                next_obs=next_obs,
                bs=bs,
                task_horizon=task_horizons[i]
                )
            rollout_buffer = task_return_dict["rollout_buffer"]
            mean_success_rate = task_return_dict["mean_success_rate"]
            success_rates_eps = task_return_dict["success_rates_eps"]
            t_z_rs = task_return_dict["z_rs"]
            t_goals = task_return_dict["goals"]

            if log_summary:
                metric_name = f"success_rate/{task}"
                print(f"Logging summary {metric_name} to run {self.wandb_run.id}")
                self.wandb_run.summary[metric_name] = mean_success_rate

            print("Success rate over {} rollouts in task {}, is {}".format(len(rollout_buffer), task, mean_success_rate))#

            rollouts_dict[task] = rollout_buffer
            success_rates_eps_dict[task] = success_rates_eps
            success_rates_dict[task] = mean_success_rate
            z_rs[task] = t_z_rs
            goals[task] = t_goals

        eval_return_dict = {
            "rollouts_dict": rollouts_dict,
            "success_rates_eps": success_rates_eps_dict,
            "success_rates": success_rates_dict,
            "z_rs": z_rs,
            "goals": goals,
            }
        return eval_return_dict

    
    #other possible functionalities:
    # - generate a bunch of goals, corresponding zrs, linear regression good?
    # - pick trajectory, imitation/goal reaching, show metrics and record vid



def setup_video(output_path, name_suffix, name_prefix, fps, name_infix=""):
    #infix= "rollout"
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{name_prefix}{name_infix}{name_suffix}.mp4")
    i = 0
    while os.path.isfile(file_path):
        i += 1
        file_path = os.path.join(output_path, f"{name_prefix}{name_infix}{name_suffix}_{i}.mp4")
    print("Record video in {}".format(file_path))
    return (
        imageio.get_writer(file_path, fps=fps, codec="h264", quality=10, pixelformat="yuv420p"), #yuv420p, yuvj422p
        file_path,
    )

path_prefix="video"
class VideoRecorder(object):
    def __init__(self, buffer, output_path, env=None, name_suffix="", name_prefix="", name_infix=""):
        self.buffer = buffer
        self.env = env if env is not None else get_default_env()
        self.output_path = os.path.join(path_prefix, output_path)
        self.name_suffix = name_suffix
        self.name_prefix = name_prefix
        self.infix = name_infix
        self.render_width = 768
        self.render_height = 512


    def record(self, ep_ids, name_suffix=None, name_prefix=None, name_infix=None, buffer=None):
        buffer = buffer if buffer is not None else self.buffer
        name_suffix = name_suffix if name_suffix is not None else self.name_suffix
        name_prefix = name_prefix if name_prefix is not None else self.name_prefix
        name_infix = name_infix if name_infix is not None else self.infix
        video, video_path = setup_video(
            self.output_path, name_suffix, name_prefix, self.env.get_fps(), name_infix
        )
        for ep_id in ep_ids:
            episode = buffer[ep_id]
            obs = self.env.reset()
            for t in range(len(episode)):
                self.env.set_GT_state(episode["env_states"][t, :])
                frame = self.env.render("rgb_array", self.render_width, self.render_height)
                video.append_data(frame)
            #maybe try without this
            if self.env.name == "FetchPickAndPlaceConstruction":
                del self.env.viewer._markers[:]
        video.close()

def old_record_imitation(rollout_base, rollouts_fb, output_path, name_suffix=""):
    combined_rollouts = rollout_base + rollouts_fb 
    recorder = VideoRecorder(combined_rollouts, output_path, env=get_default_env(), name_suffix=name_suffix, name_infix="Imit")
    recorder.record(np.arange(len(combined_rollouts)))

def record_imitation(rollout_base, rollouts_fb, output_path):
    pass

if __name__ == "__main__":
    pass