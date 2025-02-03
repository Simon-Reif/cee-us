import math
import gym
import torch
import torch.nn.functional as F
from torch.distributions.cauchy import Cauchy
from abc import ABC, abstractmethod

from mbrl.base_types import Env
from mbrl.controllers.abstract_controller import TrainableController
from mbrl.models.fb_models import ForwardMap, BackwardMap
from mbrl.rolloutbuffer import RolloutBuffer
from mbrl import allogger, torch_helpers


class ForwardBackwardController(TrainableController):
    def __init__(self, env, params, **kwargs):
        super().__init__(**kwargs)
        self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])

        self.embed_dim = params.embed_dim
        # TODO: preprocessing, change this
        self.obs_dim = env.observation_space.shape[0]
        # TODO: maybe save dict of different z_r for different tasks
        self.z_r = None
        # to refine z_r in multiple steps
        self.n_zr_estim_samples=0
        # TODO: preprocessing, change this
        self.obs_dim = env.observation_space.shape[0]

        self.forward_network = ForwardMap(obs_dim=self.obs_dim, embed_dim=self.embed_dim, action_dim=self.action_discretization.num_actions())
        self.backward_network = BackwardMap(obs_dim=self.obs_dim, embed_dim=self.embed_dim)
        # value 0.5 from paper
        self.cauchy = Cauchy(torch.tensor([0.0]), torch.tensor([0.5]))
        self.forward_target_network = ForwardMap(obs_dim=self.obs_dim, embed_dim=self.embed_dim, action_dim=self.action_discretization.num_actions())
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

    
    def set_forward_map(self, forward_map):
        self.forward_network = forward_map
    def set_backward_map(self, backward_map):
        self.backward_network = backward_map
    
    # fits F, B to data from rollout_buffer
    # metrics dict for tensorboard
    def train(self, rollout_buffer, metrics):
        pass
    
    # for calculating zr s for different tasks with same offline data
    @torch.no_grad()
    def calculate_Bs(self, observation_list: torch.Tensor)->torch.Tensor:
        #observation_list=torch.tensor(observation_list, device=torch_helpers.device)
        bs=self.backward_network(observation_list)
        return bs

    @torch.no_grad()
    def estimate_z_r(self, obs, actions, next_obs, env: Env, bs=None, wr=True):
        if bs is None:
            bs = self.calculate_Bs(next_obs)
        rewards = -env.cost_fn(obs, actions, next_obs).to(torch_helpers.device)
        if wr:
            rewards = rewards*F.softmax(10 * rewards, dim=0)
        z_r=torch.matmul(rewards.T, bs)
        return self.project_z(z_r)


    def project_z(self, z):
        return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)


    # fits z_r to compute_reward() given by env using data from rollout_buffer
    # see FB paper 23
    # TODO: maybe parameter for size of buffer to use
    # TODO: next obs instead of obs
    def policy_parameter_estimation(self, env, rollout_buffer, start_from_current_zr=False):
        #TODO: fit this to rollout_buffer specification
        with torch.no_grad():
            obs=torch.tensor(rollout_buffer.as_array("observations"))
            obs=torch.flatten(obs, end_dim=1)
            b_obs=self.backward_network(obs)
            sum_representations = torch.sum([self.backward_network(obs) * env.compute_reward(obs) for obs in range(rollout_buffer.flat)], dim=0)
            # TODO maybe change for dict of z_rs for different tasks
            if start_from_current_zr:
                if self.z_r is None:
                    self.z_r = 0
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
        
'''
RolloutBuffer:
list of rollouts : dict of obs, act, etc : list of transitions : dimension of obs, act, etc
'''

# test loading, saving
# test input, output dims
# test zr estimation
if __name__ == "__main__":
    pass