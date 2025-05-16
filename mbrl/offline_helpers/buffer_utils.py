import copy
import pickle
import os

import numpy as np
import smart_settings
import yaml



from mbrl.environments import env_from_string
from mbrl.environments.abstract_environments import MaskedGoalSpaceEnvironmentInterface
from mbrl.rolloutbuffer import Rollout, RolloutBuffer


def save_buffer(buffer, save_path):
    dir = os.path.dirname(save_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(save_path, "wb") as f:
        pickle.dump(buffer, f)

def load_buffer(path):
    with open(path, 'rb') as f:
        buffer = pickle.load(f)
    return buffer

def filter_buffer_by_length(buffer, max_length=None, save_path=None):
    new_rollouts = [copy.deepcopy(rollout) for rollout in buffer if len(rollout) <= max_length]
    new_buffer = RolloutBuffer(rollouts=new_rollouts)
    if save_path is not None:
        save_buffer(new_buffer, save_path)
    return new_buffer


# indices of the last timestep of each episode
def truncate_episodes(buffer: RolloutBuffer, indices, save_path=None):
    new_buffer = copy.deepcopy(buffer)
    for i in range(len(buffer)):
        new_buffer[i]._data = buffer[i]._data[:indices[i]+1]
    if save_path is not None:
        save_buffer(new_buffer, save_path)
    return new_buffer


def get_buffer_wo_goals(buffer:RolloutBuffer, env: MaskedGoalSpaceEnvironmentInterface):
    #rollouts = copy.deepcopy(buffer.rollouts)
    rollouts = buffer.rollouts
    field_names = buffer[0].field_names()
    new_rollouts = []
    for rollout in rollouts:
        new_dict = {field_name: copy.deepcopy(rollout[field_name]) for field_name in field_names}
        new_dict["observations"] = env.observation_wo_goal(rollout["observations"])
        new_dict["next_observations"] = env.observation_wo_goal(rollout["next_observations"])
        new_rollout = Rollout.from_dict(**new_dict)
        new_rollouts.append(new_rollout)
    return RolloutBuffer(rollouts=new_rollouts)

def add_goals_to_buffer(buffer:RolloutBuffer, env: MaskedGoalSpaceEnvironmentInterface):
    rollouts = buffer.rollouts
    field_names = buffer[0].field_names()
    new_rollouts = []
    for rollout in rollouts:
        new_dict = {field_name: copy.deepcopy(rollout[field_name]) for field_name in field_names}
        goal = env.goal_from_state(rollout["env_states"][0])
        new_dict["observations"] = env.append_goal_to_observation(rollout["observations"], goal)
        new_dict["next_observations"] = env.append_goal_to_observation(rollout["next_observations"], goal)
        new_rollout = Rollout.from_dict(**new_dict)
        new_rollouts.append(new_rollout)
    return RolloutBuffer(rollouts=new_rollouts)

def load_buffer_with_goals(params, dir=None):
    if dir is not None:
        dir = dir
    else:
        dir = params.training_data_dir
    buffer = load_buffer(os.path.join(dir, 'rollouts'))
    return buffer


def load_buffer_wog(params, dir=None):
    if dir is not None:
        dir = dir
    else:
        dir = params.training_data_dir
    wog_path=os.path.join(dir,'rollouts_wog')
    if os.path.exists(wog_path):
        #print("Loading existing buffer without goals")
        buffer = load_buffer(wog_path)
    else:
        print("Extracting observations without goals and saving buffer")
        raw_buffer = load_buffer(os.path.join(dir, 'rollouts'))
        env = env_from_string(params.env, **params.env_params)
        buffer = get_buffer_wo_goals(raw_buffer, env)
        save_buffer(buffer, wog_path)
    return buffer
    

def combine_buffers(buffer_1, buffer_2, save_path=None):
    new_rollouts = copy.deepcopy(buffer_1.rollouts)
    new_rollouts.extend(buffer_2.rollouts)
    # new_rollouts is "_CustomList" type, so we need to convert it to a list
    new_buffer = RolloutBuffer(rollouts=new_rollouts._list)
    if save_path is not None:
        dir = os.path.dirname(save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(save_path, "wb") as f:
            pickle.dump(new_buffer, f)
    return new_buffer

def combine_buffer_list(buffer_list, save_path=None):
    new_rollouts = copy.deepcopy(buffer_list[0].rollouts)
    for i in range(1, len(buffer_list)):
        new_rollouts.extend(buffer_list[i].rollouts)
    new_buffer = RolloutBuffer(rollouts=new_rollouts._list)
    if save_path is not None:
        dir = os.path.dirname(save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(save_path, "wb") as f:
            pickle.dump(new_buffer, f)
    return new_buffer


def combine_buffers_from_dirs(path_1, path_2, path_combined=None):
    with open(path_1, 'rb') as f:
        buffer_1 = pickle.load(f)
    with open(path_2, 'rb') as f:
        buffer_2 = pickle.load(f)
    new_buffer = combine_buffers(buffer_1, buffer_2, save_path=path_combined)
    return new_buffer


# freeplay and planners return rollout buffers with different types
# to combine them the type of "successes" has to be changed
def repair_dtype_bug(buffer, save_path=None):
    rollouts = buffer.rollouts
    field_names = buffer[0].field_names()
    new_rollouts = []
    for rollout in rollouts:
        new_dict = {field_name: copy.deepcopy(rollout[field_name]) for field_name in field_names}
        suc = rollout["successes"]
        if suc.ndim > 1:
            new_dict["successes"] = rollout["successes"].squeeze(axis=1)
        new_rollout = Rollout.from_dict(**new_dict)
        new_rollouts.append(new_rollout)
    new_buffer = RolloutBuffer(rollouts=new_rollouts)
    if save_path is not None:
        dir = os.path.dirname(save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(save_path, "wb") as f:
            pickle.dump(new_buffer, f)
    return new_buffer

# min_successses : {1,2}
# task: flip, throw behave the same 
#       pp, stack specific cutoff conditions
def process_planner_buffer(working_dir, buffer_dir, min_successes=2, max_length=99):
    stable_T = 5 # as in "calculate success rates"
    trunc_subdir = os.path.join(buffer_dir, 'truncated')   
    os.makedirs(trunc_subdir, exist_ok=True)
    filtered_subdir = os.path.join(buffer_dir, 'filtered')
    os.makedirs(filtered_subdir, exist_ok=True)
    params = smart_settings.load(os.path.join(working_dir, 'settings.json'), make_immutable=True)
    env = env_from_string(params.env, **params["env_params"])

    
    with open(os.path.join(buffer_dir, "rollouts"), 'rb') as f:
        buffer_with_goals = pickle.load(f)
    buffer_with_goals = repair_dtype_bug(buffer_with_goals)
    buffer_wog = get_buffer_wo_goals(buffer_with_goals, env)
    with open(os.path.join(buffer_dir, "rollouts_wog"), "wb") as f:
                    pickle.dump(buffer_wog, f)
    stats_dict = {}
    eps_successes = []
    first_successes_one_block = []
    first_successes_two_blocks = []
    indices=[]
    for i in range(len(buffer_with_goals)):
        timesteps_successes = env.eval_success(buffer_with_goals[i]["next_observations"])
        vals, unique_indices, unique_counts = np.unique(timesteps_successes, return_index=True, return_counts=True)
        
        # independent of tasks
        idcs_ones = vals==1
        if any(idcs_ones):
            first_succ_one_block = unique_indices[idcs_ones][0]
            first_successes_one_block.append(first_succ_one_block)
        idcs_twos = vals==2
        if any(idcs_twos):
            first_succ_two_blocks = unique_indices[idcs_twos][0]
            first_successes_two_blocks.append(first_succ_two_blocks)
        
        # TODO: change this part for other tasks than Throw&
        if "tower" in env.case:
            dy = np.diff(timesteps_successes)
            stable = np.logical_and(timesteps_successes[1:]==env.num_blocks, dy==0)
            success = np.sum(stable)>stable_T

            stable_indices = np.nonzero(stable)[0]
            if success:
                indices.append(stable_indices[stable_T])
            else:
                indices.append(len(timesteps_successes))
            eps_successes.append(success)
            
        elif env.case == 'PickAndPlace':
            successes = timesteps_successes==min_successes
            indices_successes = np.nonzero(successes)[0]
            if len(indices_successes) <= stable_T:
                indices.append(len(timesteps_successes))
            else:
                indices.append(indices_successes[stable_T])

            eps_successes.append(vals[unique_counts>stable_T][-1]/env.nObj)

        else: # Flip, Throw
            if min_successes in vals:
                timestep = unique_indices[vals==min_successes][0]
                indices.append(timestep)
            else:
                indices.append(len(timesteps_successes))
            # as in "calculate success rates" function
            eps_successes.append(timesteps_successes[-1]/env.nObj)
        
    eps_successes = np.array(eps_successes)
    mean_success, std_successes = eps_successes.mean(), eps_successes.std()
    first_successes_one_block = np.array(first_successes_one_block)
    first_successes_two_blocks = np.array(first_successes_two_blocks)
    mean_first_success_one_block = first_successes_one_block.mean() if len(first_successes_one_block) > 0 else None
    mean_first_success_two_blocks = first_successes_two_blocks.mean() if len(first_successes_two_blocks) > 0 else None
    num_success_one_block = len(first_successes_one_block)
    num_success_two_blocks = len(first_successes_two_blocks)
    stats_dict = {"mean_success": mean_success.item(),
                  "std_success": std_successes.item(),
                  "mean_first_success_one_block": mean_first_success_one_block.item(),
                  "mean_first_success_two_blocks": mean_first_success_two_blocks.item(),
                  "num_success_one_block": num_success_one_block,
                  "num_success_two_blocks": num_success_two_blocks,}
    

    buffer_trunc=truncate_episodes(buffer_wog, indices, save_path=os.path.join(trunc_subdir, "rollouts_wog"))
    buffer_filtered=filter_buffer_by_length(buffer_trunc, max_length=max_length,
                                            save_path=os.path.join(filtered_subdir, "rollouts_wog"))
    num_transitions = buffer_filtered.get_lengths_rollouts().sum()
    stats_dict.update({"filtered_buffer_length": len(buffer_filtered),
                       "num_transitions": num_transitions})
    print(f"Processed Buffer from {buffer_dir}")
    dict_pathname=os.path.join(buffer_dir, "stats.yaml")
    with open(dict_pathname, 'w') as f:
        yaml.dump(stats_dict, f, default_flow_style=False, sort_keys=False)
    print(f"Stats {stats_dict}")

# selection: {random, first, last}
def sub_buffer(buffer, num_episodes, selection="random", save_path=None):
    if selection=="random":
        new_buffer = buffer.random_n_rollouts(num_episodes)
    elif selection=="first":
        new_buffer = buffer.first_n_rollouts(num_episodes)
    elif selection=="last":
        new_buffer = buffer.last_n_rollouts(num_episodes)
    else:
        raise ValueError(f"Unknown selection variant: {selection}")
    if save_path:
        save_buffer(new_buffer, save_path)
    return new_buffer

def yaml_load(path):
    with open(path, 'r') as f:
        data_dict = yaml.safe_load(f)
    return data_dict

def yaml_save(data_dict, path):
    with open(path, 'w') as f:
        yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)

def update_yaml(path, update_dict):
    if os.path.exists(path):
        data_dict = yaml_load(path)
        data_dict.update(update_dict)
    else:
        data_dict = update_dict
    yaml_save(data_dict, path)

# add "rollouts" (with goals) to truncated and filtered buffer dirs 
if __name__=="__main__":
    working_dirs = ["results/cee_us/zero_shot/2blocks/225iters/flip_4500/gnn_ensemble_icem",
                "results/cee_us/zero_shot/2blocks/225iters/pp_4500/gnn_ensemble_icem",
                "results/cee_us/zero_shot/2blocks/225iters/stack_4500/gnn_ensemble_icem",
                "results/cee_us/zero_shot/2blocks/225iters/throw_4500/gnn_ensemble_icem"]
    for working_dir in working_dirs:
        params = smart_settings.load(os.path.join(working_dir, 'settings.json'), make_immutable=True)
        env = env_from_string(params.env, **params["env_params"])
        filt_dir = os.path.join(working_dir, "checkpoints_000", "filtered")
        buffer_wog = load_buffer(os.path.join(filt_dir, "rollouts_wog"))
        buffer_with_goals = add_goals_to_buffer(buffer_wog, env)
        save_buffer(buffer_with_goals, os.path.join(filt_dir, "rollouts"))
        
    

if False and __name__=="__main__":
    freeplay_path = "results/cee_us/construction/2blocks/gnn_ensemble_cee_us_freeplay/checkpoints_225/rollouts_wog"
    flip_path = "datasets/construction/planner/filtered/1500/flip/rollouts_wog"
    comb_path = "datasets/construction/freeplay_plus/flip/rollouts_wog"

    flip_buffer = load_buffer(flip_path)
    
    num_transitions = flip_buffer.get_lengths_rollouts().sum()
    print(f"Filtered buffer contains {num_transitions} transitions.")
    flip_dir = os.path.dirname(flip_path)
    dict_path = os.path.join(flip_dir, "stats.yaml")
    update_yaml(dict_path, {"num_transitions": num_transitions})

    freeplay_buffer = load_buffer(freeplay_path)
    comb_buffer = combine_buffers(freeplay_buffer, flip_buffer, save_path=comb_path)
    print(f"Combined buffer contains {len(comb_buffer)} rollouts, saved at {comb_path}")




if False and __name__=="__main__":
    buffer_path = "results/cee_us/zero_shot/2blocks/225iters/flip_4500/gnn_ensemble_icem/checkpoints_000/filtered/rollouts_wog"
    target_path ="datasets/construction/planner/filtered/1500/flip/rollouts_wog"
    buffer = load_buffer(buffer_path)
    subbuffer = sub_buffer(buffer, num_episodes=1500, selection="random", save_path=target_path)
    print(f"Saved sub buffer of {buffer_path} to {target_path}")
    print(f"Sub buffer contains {len(subbuffer)} rollouts")

if False and __name__== "__main__":
    # working_dir = "results/cee_us/zero_shot/2blocks/225iters/flip_4500/gnn_ensemble_icem"
    # buffer_dir = os.path.join(working_dir, "checkpoints_000")
    # process_planner_buffer(working_dir, buffer_dir, min_successes=2, max_length=99)

    working_dirs = ["results/cee_us/zero_shot/2blocks/225iters/pp_4500/gnn_ensemble_icem",
                "results/cee_us/zero_shot/2blocks/225iters/stack_4500/gnn_ensemble_icem",
                "results/cee_us/zero_shot/2blocks/225iters/throw_4500/gnn_ensemble_icem"]
    buffer_dirs = [os.path.join(dir, "checkpoints_000") for dir in working_dirs]
    for working_dir, buffer_dir in zip(working_dirs, buffer_dirs):
        process_planner_buffer(working_dir, buffer_dir, min_successes=2, max_length=99)
    


if False and __name__ == "__main__":
    import yaml
    paths_trunc = ["datasets/construction/bc/truncated/flip/rollouts_wog",
                    "datasets/construction/bc/truncated/flip_2/rollouts_wog",
                   "datasets/construction/bc/truncated/flip_3/rollouts_wog",]
    target_dirs = ["datasets/construction/filtered/flip",
                   "datasets/construction/filtered/flip_2",
                   "datasets/construction/filtered/flip_3",]
    for path_trunc, target_dir in zip(paths_trunc, target_dirs):
        with open(path_trunc, 'rb') as f:
            buffer_trunc = pickle.load(f)
        buffer_filtered = filter_buffer_by_length(buffer_trunc, max_length=99)
        save_buffer(buffer_filtered, os.path.join(target_dir, "rollouts_wog"))
        num_eps = len(buffer_filtered)
        length_eps = buffer_filtered.get_lengths_rollouts()
        mean_length = np.mean(length_eps)
        print(f"Filtered buffer contains {num_eps} rollouts.")
        print(f"Mean length of rollouts: {mean_length}")
        meta_dict = {"num_eps": num_eps, "mean_length": mean_length}
        dict_pathname=os.path.join(target_dir, "stats.yaml")
        with open(dict_pathname, 'w') as f:
            yaml.dump(meta_dict, f, default_flow_style=False, sort_keys=False)

if False and __name__ == "__main__":
    import smart_settings
    from mbrl.environments import env_from_string
    dirs = ["results/cee_us/zero_shot/2blocks/225iters/construction_flip_2/gnn_ensemble_icem",
            "results/cee_us/zero_shot/2blocks/225iters/construction_flip_3/gnn_ensemble_icem",]
    target_paths = ["datasets/construction/bc/truncated/flip_2/rollouts_wog",
                   "datasets/construction/bc/truncated/flip_3/rollouts_wog",]
    params = smart_settings.load(os.path.join(dirs[0], 'settings.json'), make_immutable=True)
    env = env_from_string(params.env, **params["env_params"])
    buff_dirs = [os.path.join(dir, 'checkpoints_000') for dir in dirs]
    for buff_dir, target_path in zip(buff_dirs, target_paths):
        with open(os.path.join(buff_dir, "rollouts"), 'rb') as f:
            buffer_with_goals = pickle.load(f)
        indices = []
        for i in range(len(buffer_with_goals)):
            rollout_success = env.eval_success(buffer_with_goals[i]["next_observations"])
            vals, unique_indices = np.unique(rollout_success, return_index=True)
            if 2 in vals:
                indices.append(unique_indices[vals==2][0])
            else:
                indices.append(len(rollout_success))
        print("Indices of last timestep of each episode: ", indices)
        with open(os.path.join(buff_dir, "rollouts_wog"), 'rb') as f:
            buffer_wog = pickle.load(f)
        truncate_episodes(buffer_wog, indices, save_path=target_path)


if False and __name__ == "__main__":
    target_path="datasets/construction/fb/freeplay_plus_planners/rollouts_wog"
    task_strings = ["flip", "throw", "pp", "stack"]
    dirs_panner = [f"results/cee_us/zero_shot/2blocks/225iters/construction_{task}/gnn_ensemble_icem/checkpoints_000/rollouts_wog" 
                   for task in task_strings]
    dir_fp = "results/cee_us/construction/2blocks/gnn_ensemble_cee_us_freeplay/checkpoints_225/rollouts_wog"
    dirs = [dir_fp] + dirs_panner
    buffer_list=[]
    for dir in dirs:
        with open(dir, 'rb') as f:
            buffer = pickle.load(f)
            buffer_list.append(buffer)
    buffer_comb = combine_buffer_list(buffer_list, save_path=target_path)
    print("Combined buffer contains {} rollouts, each with length {}".format(len(buffer_comb), buffer_comb[0]["observations"].shape[0]))
    print(f"Combined buffer saved to {target_path}")
    
if False and __name__=="__main__":
    filename="rollouts_wog"
    import smart_settings
    from mbrl.environments import env_from_string
    working_dir = f'results/cee_us/zero_shot/2blocks/225iters/construction_flip_3/gnn_ensemble_icem'
    params = smart_settings.load(os.path.join(working_dir, 'settings.json'), make_immutable=False)
    env = env_from_string(params.env, **params["env_params"])
    buffer_dir = os.path.join(working_dir, 'checkpoints_000')
    with open(os.path.join(buffer_dir, "rollouts"), 'rb') as f:
        buffer = pickle.load(f)
    buffer = get_buffer_wo_goals(buffer, env)
    buffer = repair_dtype_bug(buffer, save_path=os.path.join(buffer_dir, "rollouts_wog"))

    path_0 = "results/cee_us/zero_shot/2blocks/225iters/construction_flip_3k/rollouts_wog"
    with open(path_0, 'rb') as f:
        buffer_0 = pickle.load(f)
    target_path = "results/cee_us/zero_shot/2blocks/225iters/construction_flip_4500/rollouts_wog"
    buffer_comb = combine_buffers(buffer_0, buffer, save_path=target_path)
        
# in lieu of commenting out
if False and __name__=="__main__":
    filename="rollouts_wog"
    import smart_settings
    from mbrl.environments import env_from_string
    working_dir = f'results/cee_us/construction/2blocks/gnn_ensemble_cee_us_freeplay'
    params = smart_settings.load(os.path.join(working_dir, 'settings.json'), make_immutable=False)
    env = env_from_string(params.env, **params["env_params"])
    with open(os.path.join(working_dir, 'checkpoints_225/rollouts_wog'), 'rb') as f:
        buffer_freeplay = pickle.load(f)

    buffer_dirs = [
        "results/cee_us/zero_shot/2blocks/225iters/construction_throw/gnn_ensemble_icem/checkpoints_000",
        "results/cee_us/zero_shot/2blocks/225iters/construction_pp/gnn_ensemble_icem/checkpoints_000",
        "results/cee_us/zero_shot/2blocks/225iters/construction_stack/gnn_ensemble_icem/checkpoints_000"]
    target_dirs = [
        "datasets/construction/fb/freeplay_plus_throw",
        "datasets/construction/fb/freeplay_plus_pp",
        "datasets/construction/fb/freeplay_plus_stack"]
    #buffer_paths = [os.path.join(buffer_dir, filename) for buffer_dir in buffer_dirs]
    target_paths = [os.path.join(target_dir, filename) for target_dir in target_dirs]
    
    for buffer_dir, target_path in zip(buffer_dirs, target_paths):
        with open(os.path.join(buffer_dir, "rollouts"), 'rb') as f:
            buffer = pickle.load(f)
        buffer = get_buffer_wo_goals(buffer, env)
        buffer = repair_dtype_bug(buffer, save_path=os.path.join(buffer_dir, "rollouts_wog"))
        buffer = combine_buffers(buffer_freeplay, buffer, save_path=target_path)
        print("Combined buffer contains {} rollouts, each with length {}".format(len(buffer), buffer[0]["observations"].shape[0]))
        print(f"Combined buffer saved to {target_path}")