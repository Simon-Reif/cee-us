import numpy as np
from mbrl.offline_helpers.buffer_utils import load_buffer_with_goals, load_buffer_wog


#dicts defining with datasets count as "expert" data for tasks
dataset_to_task_map = {
    "planner_flip": "Flip",
    "planner_throw": "Throw",
    "planner_pp": "PickAndPlace",
    "planner_stack": "Singletower",
}

# in future values may be lists
task_to_dataset_map = {
    "Flip": "planner_flip",
    "Throw": "planner_throw",
    "PickAndPlace": "planner_pp",
    "Singletower": "planner_stack",
}

# data: n_datasets x dataset length(heterogenous) x data dim
# probs: n_datasets
def mean_std_weighted_data(data, probs):
    means_individual = [np.mean(datum, axis=0) for datum in data]
    mean = np.array(means_individual).T @ probs
    vars_individual = [((datum - mean)**2).sum(axis=0) / len(datum) for datum in data]
    #this only works in later versions of numpy
    # vars_individual = [np.var(datum, axis=0, mean=mean) for datum in data]
    var = np.array(vars_individual).T @ probs
    std = np.sqrt(var)
    return mean, std

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
            if "with_goals" in self.training_data[name] and self.training_data[name]["with_goals"]:
                print(f"Loading buffer with goals from {dir}")
                buffer = load_buffer_with_goals(dir=dir)
            else:
                buffer = load_buffer_wog(dir=dir)
            max_eps = self.training_data[name]["max_episodes"]
            if max_eps and len(buffer) > max_eps:
                buffer = buffer.random_n_rollouts(num_rollouts=max_eps)
                if self.debug:
                    print(f"Shortened buffer {name} to {len(buffer)}")
            sum_weights+=self.training_data[name]["weight"]
            print(f"Dataset {name}: loaded {len(buffer)} episodes from {dir}")
            self.buffers.append(buffer)
        self.probs = np.array([self.training_data[name]["weight"]/sum_weights for name in self.names])

    def get_mean_std(self, recalculate=True):
        if not recalculate and hasattr("mean"):
            return self.mean, self.std
        buffer_obs = [buffer["next_observations"] for buffer in self.buffers]
        buffer_obs = np.array(buffer_obs, dtype=object)
        mean, std = mean_std_weighted_data(buffer_obs, self.probs)
        # if self.debug:
        #     print(f"Mean: {mean}, Std: {std}")
        self.mean = mean
        self.std = std
        return mean, std
        
    #TODO: maybe option to sample according to buffer size?
    # work like buffer from outside
    def sample(self, num_samples):
        # this is probably slow but first thing that came to mind for sampling categorically
        selection = np.random.choice(len(self.buffers), size=num_samples, p=self.probs)
        vals, counts = np.unique(selection, return_counts=True)
        # if self.debug:
        #     print(f"Vals: {vals}, Counts: {counts}")
        batches = [self.buffers[val].sample(count)  for val, count in zip(vals, counts)]
        batch = np.concatenate(batches)
        return batch
    
    def sample_start_states(self, num_samples):
        selection = np.random.choice(len(self.buffers), size=num_samples, p=self.probs)
        vals, counts = np.unique(selection, return_counts=True)
        # if self.debug:
        #     print("Sampling Start States:")
        #     print(f"Vals: {vals}, Counts: {counts}")
        start_states = [self.buffers[val].sample_start_states(count)  for val, count in zip(vals, counts)]
        start_states = np.concatenate(start_states)
        return start_states
        
    def get_obs_dim(self):
        return self.buffers[0][0]["observations"].shape[-1]
    def get_action_dim(self):
        return self.buffers[0][0]["actions"].shape[-1]
    
    # returns a list of tuples (task, trajectories)
    def get_tasks_and_traj_from_exp_data(self, num_samples):
        return_tuples = []
        for idx, data_name in enumerate(self.names):
            if data_name in dataset_to_task_map:
                task = dataset_to_task_map[data_name]
                exp_data = self.buffers[idx]
                trajectories = exp_data.random_n_rollouts(num_samples)
                return_tuples.append((task, trajectories))
        return return_tuples

    def maybe_get_expert_buffer(self, task):
        if task in task_to_dataset_map:
            dataset = task_to_dataset_map[task]
            if dataset in self.names:
                idx = self.names.index(dataset)
                return self.buffers[idx]
        return False


if __name__ == "__main__":
    ar1 = np.random.randint(0, 19, size=(10, 42))
    ar2 = np.random.randint(0, 15, size=(10, 42))
    ar3 = np.random.randint(0, 9, size=(15, 42)) # *2

    probs = np.array([0.2, 0.2, 0.6])

    buff_arr = np.array([ar1, ar2, ar3], dtype=object)
    print(buff_arr.shape)

    mean_bm, std_bm = mean_std_weighted_data(buff_arr, probs)

    ar3_double = np.tile(ar3, (2, 1))

    arr_concat = np.concatenate([ar1, ar2, ar3_double], axis=0)
    mean_concat = np.mean(arr_concat, axis=0)
    std_concat = np.std(arr_concat, axis=0)
    print("Mean BM:", mean_bm)  
    print("Mean Concat:", mean_concat)
    print("Std BM:", std_bm)
    print("Std Concat:", std_concat)
    