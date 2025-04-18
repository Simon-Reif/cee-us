import copy
import pickle
import os



from mbrl.environments.abstract_environments import MaskedGoalSpaceEnvironmentInterface
from mbrl.rolloutbuffer import Rollout, RolloutBuffer


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


def load_buffer_wog(params):
    wog_path=os.path.join(params.training_data_dir,'rollouts_wog')
    if os.path.exists(wog_path):
        print("Loading existing buffer without goals")
        with open(os.path.join(params.training_data_dir, 'rollouts_wog'), 'rb') as f:
            buffer = pickle.load(f)
    else:
        print("Extracting observations without goals and saving buffer")
        with open(os.path.join(params.training_data_dir, 'rollouts'), 'rb') as f:
            raw_buffer = pickle.load(f)
        #TODO: pick out obs without goals, save
        env = env_from_string(params.env, **params.env_params)
        buffer = get_buffer_wo_goals(raw_buffer, env)
        with open(wog_path, "wb") as f:
                    pickle.dump(buffer, f)
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


if __name__=="__main__":
    filename="rollouts_wog"
    import smart_settings
    from mbrl.environments import env_from_string
    working_dir = f'results/cee_us/zero_shot/2blocks/225iters/construction_flip_2/gnn_ensemble_icem'
    params = smart_settings.load(os.path.join(working_dir, 'settings.json'), make_immutable=False)
    env = env_from_string(params.env, **params["env_params"])
    buffer_dir = os.path.join(working_dir, 'checkpoints_000')
    with open(os.path.join(buffer_dir, "rollouts"), 'rb') as f:
        buffer = pickle.load(f)
    buffer = get_buffer_wo_goals(buffer, env)
    buffer = repair_dtype_bug(buffer, save_path=os.path.join(buffer_dir, "rollouts_wog"))

    path_0 = "results/cee_us/zero_shot/2blocks/225iters/construction_flip/gnn_ensemble_icem/checkpoints_000/rollouts_wog"
    with open(path_0, 'rb') as f:
        buffer_0 = pickle.load(f)
    target_path = "results/cee_us/zero_shot/2blocks/225iters/construction_flip_3k/rollouts_wog"
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