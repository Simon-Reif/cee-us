#adapts https://github.com/facebookresearch/metamotivo
import copy
import math
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

from mbrl.environments.fpp_construction_env import FetchPickAndPlaceConstruction
from mbrl.models.fb_model import FBModel
from mbrl.models.fb_nn import _soft_update_params, eval_mode, weight_init
from mbrl.offline_helpers.buffer_manager import BufferManager
from mbrl import torch_helpers


# no longer inherits from "Controller" or "TrainableController" since inherent cost_fn, train() method 
# and env don't make sense
class ForwardBackwardController():
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.z_dim = params.model.archi.z_dim
        self.obs_dim = params.model.obs_dim
        self.action_dim = params.model.action_dim
        self.z_r = None
        self._model = FBModel(params.model)
        self.params = params #controller_params
        self.setup_training()
        self.setup_compile()
        self._model.to(torch_helpers.device)
        self.data_mean = None
        self.data_std = None


    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.backward_optimizer = torch.optim.Adam(
            self._model._backward_map.parameters(),
            lr=self.params.train.lr_b,
            #capturable=self.params.cudagraphs and not self.params.compile,
            eps=self.params.train.eps,
            weight_decay=self.params.train.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            self._model._forward_map.parameters(),
            lr=self.params.train.lr_f,
            #capturable=self.params.cudagraphs and not self.params.compile,
            eps=self.params.train.eps,
            weight_decay=self.params.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.params.train.lr_actor,
            #capturable=self.params.cudagraphs and not self.params.compile,
            eps=self.params.train.eps,
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

    @torch.no_grad()
    def sample_mixed_z(self, train_goal = None, *args, **kwargs):
    #def sample_mixed_z(self, train_goal: torch.Tensor | None = None, *args, **kwargs):
        # samples a batch from the z distribution used to update the networks
        z = self._model.sample_z(self.params.train.batch_size, device=torch_helpers.device)

        if train_goal is not None:
            perm = torch.randperm(self.params.train.batch_size, device=torch_helpers.device)
            goals = self._model._backward_map(train_goal[perm])
            goals = self._model.project_z(goals)
            mask = torch.rand((self.params.train.batch_size, 1), device=torch_helpers.device) < self.params.train.train_goal_ratio
            z = torch.where(mask, goals, z)
        return z


    def update(self, buffer_manager: BufferManager, step: int) -> Dict[str, torch.Tensor]:
        batch = buffer_manager.sample(self.params.train.batch_size)

        obs, action, next_obs, terminated = (
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["dones"],
        )
        discount = self.params.train.discount * ~terminated.astype(np.bool)
        obs, action, next_obs, discount = torch_helpers.to_tensor_device(obs, action, next_obs, discount)

        # print(f"discount: {discount}")
        # print(f"obs: {obs[0]}")
        # print(f"action: {action[0]}")
        # print(f"next_obs: {next_obs[0]}")


        if self.params.model.norm_with_entire_buffer:
            obs = self.maybe_normalize_obs(obs)
            next_obs = self.maybe_normalize_obs(next_obs)
        else:
            self._model._obs_normalizer(obs)
            self._model._obs_normalizer(next_obs)
            with torch.no_grad(), eval_mode(self._model._obs_normalizer):
                obs, next_obs = self._model._obs_normalizer(obs), self._model._obs_normalizer(next_obs)

        
        #torch.compiler.cudagraph_mark_step_begin()
        z = self.sample_mixed_z(train_goal=next_obs).clone()
        #self.z_buffer.add(z)

        q_loss_coef = self.params.train.q_loss_coef if self.params.train.q_loss_coef > 0 else None
        clip_grad_norm = self.params.train.clip_grad_norm if self.params.train.clip_grad_norm > 0 else None

        #torch.compiler.cudagraph_mark_step_begin()
        metrics = self.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            goal=next_obs,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
        )
        metrics.update(
            self.update_actor(
                obs=obs,
                action=action,
                z=z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.params.train.fb_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.params.train.fb_target_tau)

        return metrics

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef = None,
        clip_grad_norm = None,
        #q_loss_coef: float | None,
        #clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            dist = self._model._actor(next_obs, z, self._model.params.actor_std)
            next_action = dist.sample(clip=self.params.train.stddev_clip)
            target_Fs = self._model._target_forward_map(next_obs, z, next_action)  # num_parallel x batch x z_dim
            target_B = self._model._target_backward_map(goal)  # batch x z_dim
            target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
            _, _, target_M = self.get_targets_uncertainty(target_Ms, self.params.train.fb_pessimism_penalty)  # batch x batch

        # compute FB loss
        Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
        B = self._model._backward_map(goal)  # batch x z_dim
        Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

        diff = Ms - discount * target_M  # num_parallel x batch x batch
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        # compute orthonormality loss for backward embedding
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.params.train.ortho_coef * orth_loss

        q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
        if q_loss_coef is not None:
            with torch.no_grad():
                next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.params.train.fb_pessimism_penalty)  # batch
                cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                inv_cov = torch.inverse(cov)  # z_dim x z_dim
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=-1)  # batch
                target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                expanded_targets = target_Q.expand(Fs.shape[0], -1)
            Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
            q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
            fb_loss += q_loss_coef * q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm = None,
        #clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        return self.update_td3_actor(obs=obs, z=z, clip_grad_norm=clip_grad_norm)

    def update_td3_actor(self, obs: torch.Tensor, z: torch.Tensor, clip_grad_norm = None) -> Dict[str, torch.Tensor]:
    #def update_td3_actor(self, obs: torch.Tensor, z: torch.Tensor, clip_grad_norm: float | None) -> Dict[str, torch.Tensor]:
        dist = self._model._actor(obs, z, self._model.params.actor_std)
        action = dist.sample(clip=self.params.train.stddev_clip)
        Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
        Qs = (Fs * z).sum(-1)  # num_parallel x batch
        _, _, Q = self.get_targets_uncertainty(Qs, self.params.train.actor_pessimism_penalty)  # batch
        actor_loss = -Q.mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "q": Q.mean().detach()}

    # def get_targets_uncertainty(
    #     self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc


    def set_zr(self, z_r):
        self.z_r = z_r


    @torch.no_grad()
    def maybe_normalize_obs(self, obs):
        if self.params.model.norm_obs:
            obs = (obs-self.data_mean)/self.data_std
        return obs
    
    def maybe_add_batch_dim(self, obs):
        if obs is not None and obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs
    
    # for calculating zr s for different tasks with same offline data
    # TODO: with batch norm if norm obs parameter; separate function for individual states
    # TODO: batch norm OR normalize with data mean, std?
    @torch.no_grad()
    def calculate_Bs(self, next_obs: torch.Tensor)->torch.Tensor:
        next_obs = self.maybe_add_batch_dim(next_obs)
        next_obs = self.maybe_normalize_obs(next_obs)
        bs=self._model._backward_map(next_obs)
        return bs

    def _zr_from_bs_rews(self, bs, rews, wr=True):
        # bs: batch x z_dim
        # rews: batch
        rews = torch_helpers.to_tensor(rews).to(torch_helpers.device)
        if wr:
            rews = rews * F.softmax(10 * rews, dim=0)
        z_r = torch.matmul(rews.T, bs)
        return self.project_z(z_r)
        
    # only works with batches for now
    @torch.no_grad()
    def estimate_z_r(self, next_obs, goal, env: FetchPickAndPlaceConstruction, bs=None, wr=True):
        if bs is None:
            bs = self.calculate_Bs(next_obs)
        #print(f"Goals for inference: {goal} type: {type(goal)}")
        rewards = env.compute_rewards_goal(torch_helpers.to_numpy(next_obs), goal)
        z_r = self._zr_from_bs_rews(bs, rewards, wr=wr)
        return z_r
    
    #TODO: next obs can be batch or single obs OR add batch dimension to use this function
    # see: Metamotivo Paper p.28
    @torch.no_grad()
    def zr_from_obs(self, next_obs=None, bs=None):
        if bs is None:
            bs = self.calculate_Bs(next_obs)
        z_r = torch.mean(bs, dim=0)
        z_r = self.project_z(z_r)
        return z_r

    def zr_from_obs_and_rews(self, next_obs=None, rewards=None, bs=None):
        if bs is None:
            bs = self.calculate_Bs(next_obs)
        z_r = self._zr_from_bs_rews(bs, rewards)
        return z_r
        
    



    def project_z(self, z):
        if self.params.model.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z


    # save forward, backward separately
    # save z_r if it exists
    def save(self, path):
        controller_dir = Path(path, "fb_controller")
        os.makedirs(controller_dir, exist_ok=True)
        model_state_dict = copy.deepcopy(self._model.state_dict())
        torch.save(model_state_dict, controller_dir/"model_state_dict.pth")
        torch.save(
            {
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "backward_optimizer": self.backward_optimizer.state_dict(),
                "forward_optimizer": self.forward_optimizer.state_dict(),
            },
            controller_dir / "optimizers.pth",
        )

    @classmethod
    def load(cls, path, params):
        controller_dir = Path(path, "fb_controller")
        agent = cls(params)
        optimizers = torch.load(str(controller_dir / "optimizers.pth"), 
                                map_location=torch_helpers.device, weights_only=True)
        agent.actor_optimizer.load_state_dict(optimizers["actor_optimizer"])
        agent.backward_optimizer.load_state_dict(optimizers["backward_optimizer"])
        agent.forward_optimizer.load_state_dict(optimizers["forward_optimizer"])
        model_state_dict = torch.load(controller_dir/"model_state_dict.pth",
                                      map_location=torch_helpers.device, weights_only=True)
        agent._model.load_state_dict(model_state_dict)
        return agent

    # only single (one dimensional) obs, no batch dimension
    @torch.no_grad()
    def get_action(self, obs, z=None, state=None, mode="test"):
        if self.z_r is None and z is None:
            raise AttributeError("z_r not set")
        if z is not None:
            z_r=z
        else:
            z_r=self.z_r
        obs = torch_helpers.to_tensor(obs).to(torch_helpers.device)
        obs = self.maybe_normalize_obs(obs)
        #only single obs should be here
        obs = obs.reshape(1, -1)
        z_r = z_r.reshape(1, -1)
        act = self.act(obs, z_r, mean=True)
        act = act.squeeze(0)
        return torch_helpers.to_numpy(act)
    
    def set_data_stats(self, data_mean, data_std):
        self.data_mean = torch_helpers.to_tensor(data_mean).to(torch_helpers.device)
        self.data_std = torch_helpers.to_tensor(data_std).to(torch_helpers.device)
        
'''
RolloutBuffer:
list of rollouts : dict of obs, act, etc : list of transitions : dimension of obs, act, etc
'''

# test loading, saving
# test input, output dims
# test zr estimation
if __name__ == "__main__":
    pass