#adapts https://github.com/facebookresearch/metamotivo
import math
import os
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

from mbrl.base_types import Env
from mbrl.controllers.abstract_controller import TrainableController, Controller
from mbrl.models.fb_model import FBModel
from mbrl.models.fb_nn import _soft_update_params, eval_mode, weight_init
from mbrl.rolloutbuffer import RolloutBuffer
from mbrl import allogger, torch_helpers


# no longer inherits from "Controller" or "TrainableController" since inherent cost_fn, train() method 
# and env don't make sense
class ForwardBackwardController():
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.logger = allogger.get_logger(scope=self.__class__.__name__, default_outputs=["tensorboard"])
        self.z_dim = params.model.archi.z_dim
        self.obs_dim = params.model.obs_dim
        self.action_dim = params.model.action_dim
        self.z_r = None
        self._model = FBModel(params.model)
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


    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        batch = replay_buffer.sample(self.params.train.batch_size)

        obs, action, next_obs, terminated = (
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["dones"],
        )
        discount = self.params.train.discount * ~terminated.astype(np.bool)
        obs, action, next_obs, discount = torch_helpers.to_tensor_device(obs, action, next_obs, discount)
   
        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs, next_obs = self._model._obs_normalizer(obs), self._model._obs_normalizer(next_obs)

        torch.compiler.cudagraph_mark_step_begin()
        z = self.sample_mixed_z(train_goal=next_obs).clone()
        #self.z_buffer.add(z)

        q_loss_coef = self.params.train.q_loss_coef if self.params.train.q_loss_coef > 0 else None
        clip_grad_norm = self.params.train.clip_grad_norm if self.params.train.clip_grad_norm > 0 else None

        torch.compiler.cudagraph_mark_step_begin()
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

    
    # for calculating zr s for different tasks with same offline data
    @torch.no_grad()
    def calculate_Bs(self, next_obs: torch.Tensor)->torch.Tensor:
        #TODO: maybe limit batch size
        #observation_list=torch.tensor(observation_list, device=torch_helpers.device)
        bs=self._model._backward_map(next_obs)
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
        if self.params.model.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z


    # save forward, backward separately
    # save z_r if it exists
    def save(self, path):
        controller_dir = os.path.join(path, "fb_controller")
        os.makedirs(controller_dir, exist_ok=True)
        
        pass

    @classmethod
    def load(self, path, params):
        pass

    @torch.no_grad()
    def get_action(self, obs, state=None, mode="train"):
        if self.z_r is None:
            raise AttributeError("z_r not set")
        obs = torch_helpers.to_tensor(obs).to(torch_helpers.device)
        act = self.act(obs, self.z_r, mean=True)
        return torch_helpers.to_numpy(act)
        
'''
RolloutBuffer:
list of rollouts : dict of obs, act, etc : list of transitions : dimension of obs, act, etc
'''

# test loading, saving
# test input, output dims
# test zr estimation
if __name__ == "__main__":
    pass