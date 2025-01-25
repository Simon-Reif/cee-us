import torch
from torch.distributions.cauchy import Cauchy

from mbrl.controllers import TrainableController
from mbrl.models.fb_models import ForwardMap, BackwardMap
from mbrl.rolloutbuffer import RolloutBuffer



class ActionDiscretization:
    def __init__(self, num_actions, action_low, action_high):
        self.num_actions = num_actions
        self.action_low = action_low
        self.action_high = action_high

    def get_action(self, action):
        return action

    def get_action_space_discrete(self):
        return self.num_actions
    
    def get_action_space_continuous(self):
        return self.disc_to_cont(self.get_action_space_discrete)
    
    def disc_to_cont(self, action):
        pass

    def get_action_low(self):
        return self.action_low

    def get_action_high(self):
        return self.action_high
    

class ForwardBackwardController(TrainableController):
    def __init__(self, env, params, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = params.embed_dim
        # TODO: preprocessing, change this
        self.obs_dim = env.observation_space.shape[0]
        self.action_discretization = ActionDiscretization()
        self.z_r = None
        # to refine z_r in multiple steps
        self.n_zr_estim_samples=0
        # TODO: preprocessing, change this
        self.obs_dim = env.observation_space.shape[0]

        self.forward_network = ForwardMap(obs_dim=self.obs_dim, embed_dim=self.embed_dim, action_dim=self.action_discretization.num_actions)
        self.backward_network = BackwardMap(obs_dim=self.obs_dim, embed_dim=self.embed_dim)
        # value 0.5 from paper
        self.cauchy = Cauchy(torch.tensor([0.0]), torch.tensor([0.5]))
        self.forward_target_network = ForwardMap(obs_dim=self.obs_dim, embed_dim=self.embed_dim, action_dim=self.action_discretization.num_actions)
        self.backward_target_network = BackwardMap(obs_dim=self.obs_dim, embed_dim=self.embed_dim)
        # load the weights into the target networks
        self.forward_target_network.load_state_dict(self.forward_network.state_dict())
        self.backward_target_network.load_state_dict(self.backward_network.state_dict())
        # if use gpu
        if params.cuda:
            self.forward_network.cuda()
            self.backward_network.cuda()
            self.forward_target_network.cuda()
            self.backward_target_network.cuda()
        # create the optimizer
        f_params = [param for param in self.forward_network.parameters()]
        b_params = [param for param in self.backward_network.parameters()]
        self.fb_optim = torch.optim.Adam(f_params + b_params, lr=params.lr)


    
    def set_action_discretization(self, action_discretization):
        self.action_discretization = action_discretization
    
    def set_forward_map(self, forward_map):
        self.forward_network = forward_map
    def set_backward_map(self, backward_map):
        self.backward_network = backward_map
    
    def train(self, rollout_buffer):
        pass

    # fits z_r to compute_reward() given by env using data from rollout_buffer
    # see FB paper 23
    # TODO: maybe parameter for size of buffer to use
    def policy_parameter_estimation(self, env, rollout_buffer, start_from_current_zr=False):
        #TODO: fit this to rollout_buffer specification
        with torch.no_grad():
            sum_representations = torch.sum([self.backward_network(obs) * env.compute_reward(obs) for obs in range(rollout_buffer.flat)], dim=0)
            if start_from_current_zr:
                self.z_r = (self.n_zr_estim_samples*self.z_r + sum_representations)/ (self.n_zr_estim_samples + rollout_buffer.size())
                self.n_zr_estim_samples += rollout_buffer.size()
            else:
                self.z_r = sum_representations / rollout_buffer.size()
                self.n_zr_estim_samples = rollout_buffer.size()

    # save forward, backward separately
    # save z_r if it exists
    # save action_discretization
    def save(self, path):
        pass

    def get_action(self, obs):
        if self.z_r is None:
            raise AttributeError("z_r not set")
        return self.get_action_from_parameter(obs, self.z_r)

    def get_action_from_parameter(self, obs, z_r):
        with torch.no_grad():
            action_discrete=torch.argmax([self.forward_network(obs, a, z_r) for a in range(self.action_discretization.get_action_space_continuous())])
            return self.action_discretization.disc_to_cont(action_discrete)
        

# test loading, saving
# test input, output dims
# test zr estimation
if __name__ == "__main__":
    pass