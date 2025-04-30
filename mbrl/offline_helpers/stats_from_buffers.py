import os
import pickle
import numpy as np
import smart_settings
import yaml
from mbrl.environments import env_from_string
from mbrl.offline_helpers.eval import calculate_success_rates


if __name__ == "__main__":
    task_strings = ["flip", "throw", "pp", "stack"]
    buffer_dirs = [f"results/cee_us/zero_shot/2blocks/225iters/construction_{task}/gnn_ensemble_icem/checkpoints_000" 
                   for task in task_strings]
    for dir in buffer_dirs:
        params = smart_settings.load(os.path.join(os.path.dirname(dir), 'settings.json'), make_immutable=False)
        env = env_from_string(params.env, **params["env_params"])
        print(f"Buffer from: {dir}")
        # rollouts instead of rolluts_wog for "calculate_success_rates" function
        buff_path = os.path.join(dir, "rollouts")
        with open(buff_path, 'rb') as f:
            buffer = pickle.load(f)
        
        success_rates_per_episode = calculate_success_rates(env, buffer)
        mean_success_csr = np.mean(success_rates_per_episode)
        print(f"Mean success at {env.case} calculated by 'calculate success_rates': {mean_success_csr}")

        num_eps_with_succ_one_block=0
        num_eps_with_succ_two_blocks=0
        for ep_id in range(len(buffer)):
            print(f"Episode {ep_id}:")
            episode = buffer[ep_id]
            successes = episode["successes"]
            # 0, 1, 2 "success" on blocks, first occurrence, counts
            u_vals, u_indices, u_counts = np.unique(successes, return_index=True, return_counts=True)
            idcs_ones = u_vals==1
            if any(idcs_ones):
                first_succ_one_block = u_indices[idcs_ones]
                num_succ_one_block = u_counts[idcs_ones]
                print(f"First success at 1 block: {first_succ_one_block}, Num: {num_succ_one_block}")
                #TODO: append to npy dict
                num_eps_with_succ_one_block += 1
            idcs_twos = u_vals==2
            if any(idcs_twos):
                first_succ_two_blocks = u_indices[idcs_twos]
                num_succ_two_blocks = u_counts[idcs_twos]
                print(f"First success at 2 blocks: {first_succ_two_blocks}, Num: {num_succ_two_blocks}")
                num_eps_with_succ_two_blocks += 1
            

        stats = {"mean_success_rate": mean_success_csr,
                "num_eps_with_succ_one_block": num_eps_with_succ_one_block,
                "num_eps_with_succ_two_blocks": num_eps_with_succ_two_blocks,}
        dict_pathname=os.join.path(dir, "stats.yaml")
        with open(dict_pathname, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False, sort_keys=False)

        #TODO: maybe save returns of unique for each episode in np object
