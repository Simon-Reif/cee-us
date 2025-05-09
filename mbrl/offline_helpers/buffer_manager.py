

import numpy as np
from mbrl.offline_helpers.buffer_utils import load_buffer_wog


class BufferManager:
    def __init__(self, params):
        self.names = params["training_data_names"]
        self.training_data = params["training_data"]
        self.norm_obs = params.controller_params.model.norm_obs
        self.debug = params.debug
        self.buffers = []
        sum_weights = 0
        for name in self.names:
            dir = self.training_data[name]["dir"]
            buffer = load_buffer_wog(params, dir=dir)
            max_eps = self.training_data[name]["max_episodes"]
            if max_eps and len(buffer) > max_eps:
                buffer = buffer.random_n_rollouts(num_rollouts=max_eps)
                if self.debug:
                    print(f"Shortened buffer {name} to {len(buffer)}")
            sum_weights+=self.training_data[name]["weight"]
            print(f"Dataset {name}: loaded {len(buffer)} episodes from {dir}")
            self.buffers.append(buffer)
        self.probs = [self.training_data[name]["weight"]/sum_weights for name in self.names]
 



    #TODO: maybe option to sample according to buffer size?

    def get_mean_std(self):
        pass

    # work like buffer from outside
    def sample(self, num_samples):
        # this is probably slow but first thing that came to mind for sampling categorically
        selection = np.random.choice(len(self.buffers), size=num_samples, p=self.probs)
        vals, counts = np.unique(selection, return_counts=True)

        if self.debug:
            print(f"Vals: {vals}, Counts: {counts}")
        batches = [self.buffers[val].sample(count)  for val, count in zip(vals, counts)]
        batch = np.concatenate(batches)
        return batch
        

    def get_obs_dim(self):
        return self.buffers[0][0]["observations"].shape[-1]
    def get_action_dim(self):
        return self.buffers[0][0]["actions"].shape[-1]