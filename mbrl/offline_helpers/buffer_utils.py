import copy

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


if __name__=="__main__":
    import os
    import numpy as np
    import pickle
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