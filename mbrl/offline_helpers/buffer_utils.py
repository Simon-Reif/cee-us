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

def combine_buffers(buffer_1, buffer_2):
    new_rollouts = copy.deepcopy(buffer_1.rollouts)
    new_rollouts.extend(buffer_2.rollouts)
    # new_rollouts is "_CustomList" type, so we need to convert it to a list
    new_buffer = RolloutBuffer(rollouts=new_rollouts._list)
    return new_buffer

def combine_buffers_from_dirs(path_1, path_2, path_combined):
    with open(path_1, 'rb') as f:
        buffer_1 = pickle.load(f)
    with open(path_2, 'rb') as f:
        buffer_2 = pickle.load(f)
    new_buffer = combine_buffers(buffer_1, buffer_2)
    dir = os.path.dirname(path_combined)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(path_combined, "wb") as f:
        pickle.dump(new_buffer, f)
    return new_buffer


# freeplay and planners return rollout buffers with different types
# to combine them the type of "successes" has to be changed
def repair_dtype_bug(buffer, save_path=None):
    rollouts = buffer.rollouts
    field_names = buffer[0].field_names()
    new_rollouts = []
    for rollout in rollouts:
        new_dict = {field_name: copy.deepcopy(rollout[field_name]) for field_name in field_names}
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
    import numpy as np
    import smart_settings
    from mbrl.environments import env_from_string
    env_name = "construction" # construction or "playground"
    working_dir = f'results/cee_us/{env_name}/2blocks/gnn_ensemble_cee_us_freeplay'

    params = smart_settings.load(os.path.join(working_dir, 'settings.json'), make_immutable=False)
    env = env_from_string(params.env, **params["env_params"])
    with open(os.path.join(working_dir, 'checkpoints_225/rollouts'), 'rb') as f:
        buffer = pickle.load(f)

    print("Buffer contains {} rollouts, each with length {}".format(len(buffer), buffer[0]["observations"].shape[0]))

    print(buffer[0]["observations"].shape)

    buffer_wog=get_buffer_wo_goals(buffer, env)
    print(buffer_wog[0].shape)