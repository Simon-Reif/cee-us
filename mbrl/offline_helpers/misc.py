
import numpy as np

# correlation or something
def cr_es_agreement_over_episodes(rewards, success_rates):
    pass

# compare compute_reward values at 0, 1 and 2 eval_success places
def cr_es_agreement(rewards, successes):
    zeroes = successes == 0
    ones = successes == 1
    twos = successes == 2
    rewards_zeroes = np.mean(rewards[zeroes]) if np.any(zeroes) else 0
    rewards_ones = np.mean(rewards[ones]) if np.any(ones) else 0
    rewards_twos = np.mean(rewards[twos]) if np.any(twos) else 0
    print(f"Mean rewards for successes 0, 0.5 and 1 are {rewards_zeroes}, {rewards_ones}, {rewards_twos}")
    return rewards_zeroes, rewards_ones, rewards_twos