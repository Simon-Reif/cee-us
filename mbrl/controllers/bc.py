import copy
import os
from pathlib import Path
from typing import Dict
import torch
from torch import nn

from mbrl import torch_helpers
from mbrl.models.fb_nn import eval_mode, weight_init
from mbrl.models.baseline_models import BCModel
from mbrl.rolloutbuffer import RolloutBuffer



class BehaviorCloningController():
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = params.model.obs_dim
        self.action_dim = params.model.action_dim
        self._model = BCModel(params.model)
        self.params = params #controller_params
        self.setup_training()
        self._model.to(torch_helpers.device)
        self.data_mean = None
        self.data_std = None


    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)  # ensure that target nets are initialized after applying the weights
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.params.train.lr_actor,
            #capturable=self.params.cudagraphs and not self.params.compile,
            eps=self.params.train.eps,
            weight_decay=self.params.train.weight_decay,
        )

    def update(self, replay_buffer: RolloutBuffer, step: int) -> Dict[str, torch.Tensor]:
        batch = replay_buffer.sample(self.params.train.batch_size)

        obs, action = (
            batch["observations"],
            batch["actions"],
        )
        obs, action = torch_helpers.to_tensor_device(obs, action)

        self._model._obs_normalizer(obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs = self._model._obs_normalizer(obs)
        clip_grad_norm = self.params.train.clip_grad_norm if self.params.train.clip_grad_norm > 0 else None

        dist = self._model._actor(obs, self._model.params.actor_std)

        loss_key = self.params.train.loss
        #theoretically the same, still testing for difference
        if loss_key == "mse":
            action_bc = dist.sample(clip=self.params.train.stddev_clip)
            bc_loss = nn.MSELoss()(action_bc, action)
        elif loss_key == "log_prob":
            likelihood = dist.log_prob(action)
            bc_loss = -likelihood.mean()
        else:
            raise ValueError(f"Unknown loss function {loss_key}")



        self.actor_optimizer.zero_grad(set_to_none=True)
        bc_loss.backward()
        clip_grad_norm = self.params.train.clip_grad_norm if self.params.train.clip_grad_norm > 0 else None
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        metrics = {"bc_loss": bc_loss.detach()}

        return metrics

    def save(self, path):
        controller_dir = Path(path, "bc_controller")
        os.makedirs(controller_dir, exist_ok=True)
        model_state_dict = copy.deepcopy(self._model.state_dict())
        torch.save(model_state_dict, controller_dir/"model_state_dict.pth")
        torch.save(
            {
                "actor_optimizer": self.actor_optimizer.state_dict(),
            },
            controller_dir / "optimizers.pth",
        )

    @classmethod
    def load(cls, path, params):
        controller_dir = Path(path, "bc_controller")
        agent = cls(params)
        optimizers = torch.load(str(controller_dir / "optimizers.pth"), 
                                map_location=torch_helpers.device, weights_only=True)
        agent.actor_optimizer.load_state_dict(optimizers["actor_optimizer"])
        model_state_dict = torch.load(controller_dir/"model_state_dict.pth",
                                      map_location=torch_helpers.device, weights_only=True)
        agent._model.load_state_dict(model_state_dict)
        return agent


    @torch.no_grad()
    def get_action(self, obs, state=None, mode="test"):
        obs = torch_helpers.to_tensor(obs).to(torch_helpers.device)
        if self.params.model.norm_obs:
            obs = (obs-self.data_mean)/self.data_std
        #only single obs should be here
        obs = obs.reshape(1, -1)
        act = self._model.act(obs, mean=True)
        act = act.squeeze(0)
        return torch_helpers.to_numpy(act)


    def set_data_stats(self, data_mean, data_std):
        self.data_mean = torch_helpers.to_tensor(data_mean).to(torch_helpers.device)
        self.data_std = torch_helpers.to_tensor(data_std).to(torch_helpers.device)


if __name__ == "__main__":
    loss = nn.MSELoss()
    input = torch.randn(3, 5)
    target = torch.randn(3, 5)
    output = -loss(input, target)
    print(output)