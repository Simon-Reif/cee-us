#adapts https://github.com/facebookresearch/metamotivo
import math
import torch
import torch.nn.functional as F
from torch.distributions.cauchy import Cauchy
from abc import ABC, abstractmethod

from mbrl.base_types import Env
from mbrl.controllers.abstract_controller import TrainableController
from mbrl.models.fb_model import FBModel
from mbrl.models.fb_nn import ForwardMap, BackwardMap, weight_init
from mbrl.rolloutbuffer import RolloutBuffer
from mbrl import allogger, torch_helpers


class ForwardBackwardController(TrainableController):
    def __init__(self, params, obs_dim, action_dim, **kwargs):
        super().__init__(**kwargs)
        self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
        self.z_dim = params.model.z_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_r = None
        self._model = FBModel(params.model, obs_dim, action_dim)
        self.params = params #controller_params
        self.setup_training()
        self.setup_compile()
        self._model.to(torch_helpers.device)


    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.backward_optimizer = torch.optim.Adam(
            self._model._backward_map.parameters(),
            lr=self.params.train.lr_b,
            capturable=self.params.cudagraphs and not self.params.compile,
            weight_decay=self.params.train.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            self._model._forward_map.parameters(),
            lr=self.params.train.lr_f,
            capturable=self.params.cudagraphs and not self.params.compile,
            weight_decay=self.params.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.params.train.lr_actor,
            capturable=self.params.cudagraphs and not self.params.compile,
            weight_decay=self.params.train.weight_decay,
        )

        # prepare parameter list
        self._forward_map_paramlist = tuple(x for x in self._model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self._model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(x for x in self._model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self._model._target_backward_map.parameters())

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.params.train.batch_size, self.params.train.batch_size, device=torch_helpers.device)
        self.off_diag_sum = self.off_diag.sum()

    
    def setup_compile(self):
        print(f"compile {self.params.compile}")
        if self.params.compile:
            mode = "reduce-overhead" if not self.params.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            self.update_fb = torch.compile(self.update_fb, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.update_actor = torch.compile(self.update_actor, mode=mode)  # use fullgraph=True to debug for graph breaks
            self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)

        print(f"cudagraphs {self.params.cudagraphs}")
        # if self.params.cudagraphs:
        #     from tensordict.nn import CudaGraphModule

        #     self.update_fb = CudaGraphModule(self.update_fb, warmup=5)
        #     self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)


    def set_zr(self, z_r):
        self.z_r = z_r
    
    # fits F, B to data from rollout_buffer
    # metrics dict for tensorboard
    def train(self, rollout_buffer, metrics):
        pass
    
    # for calculating zr s for different tasks with same offline data
    @torch.no_grad()
    def calculate_Bs(self, next_obs: torch.Tensor)->torch.Tensor:
        #observation_list=torch.tensor(observation_list, device=torch_helpers.device)
        bs=self.backward_network(next_obs)
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


    # save forward, backward separately
    # save z_r if it exists
    # save action_discretization
    def save(self, path):
        pass

    def get_action(self, obs, state=None, mode="train"):
        if self.z_r is None:
            raise AttributeError("z_r not set")
        return self.act(obs, self.z_r, mean=True)

        
'''
RolloutBuffer:
list of rollouts : dict of obs, act, etc : list of transitions : dimension of obs, act, etc
'''

# test loading, saving
# test input, output dims
# test zr estimation
if __name__ == "__main__":
    pass