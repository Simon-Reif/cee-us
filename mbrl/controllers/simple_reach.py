

# to check Reach task on mbrl/environments/fpp_construction_env.py
# this should solve it perfectly
from mbrl.params_utils import read_params_from_cmdline


class SimpleReach():

    # this could alternatively get "state" as input and extract goal from there
    def get_action(self, obs_with_goal):
        position_gripper = obs_with_goal[:3]
        position_goal = obs_with_goal[-3:]
        delta = position_goal - position_gripper
        # maybe this needs to be clipped to [-1, 1], for now try raw numbers
        action = delta.append(0)
        return action
    
if __name__ == "__main__":
    params = read_params_from_cmdline(verbose=True, save_params=False)
