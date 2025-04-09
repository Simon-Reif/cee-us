from torch import nn
import torch

from mbrl import torch_helpers
from mbrl.models.fb_nn import TruncatedNormal, eval_mode, layernorm, linear, simple_embedding


# trying to stay close to mbrl.models.fb_nn Actor
class SimpleActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_layers: int = 1) -> None:
        super().__init__()

        seq = [linear(obs_dim, hidden_dim), layernorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers-1):
            seq += [linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [linear(hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs, std):
        mu = torch.tanh(self.policy(obs))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


class BCModel(nn.Module):
    def __init__(self, params, device=None, **kwargs):
        super().__init__()
        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.params = params
        arch = params.archi
        self._actor = SimpleActor(self.obs_dim, self.action_dim, arch.actor.hidden_dim, arch.actor.hidden_layers)
        self._obs_normalizer = nn.BatchNorm1d(self.obs_dim, affine=False, momentum=0.01) if self.params.norm_obs else nn.Identity()

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        if device is not None:
            self.to(device)
        else:
            self.to(torch_helpers.device)

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        # if device is not None:
        #     self.params.device = device.type  # type: ignore
        return super().to(*args, **kwargs)

    def _normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, std: float):
        return self._actor(self._normalize(obs), std)

    def act(self, obs: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, self.params.actor_std)
        if mean:
            return dist.mean
        return dist.sample()
