#adapts https://github.com/facebookresearch/metamotivo
import copy
import math
import torch.nn.functional as F
from torch import nn
import torch

from mbrl import torch_helpers
from mbrl.models.fb_nn import build_actor, build_backward, build_forward, eval_mode


class FBModel(nn.Module):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.params = params
        arch = params.archi
        # create networks
        self._backward_map = build_backward(self.obs_dim, arch.z_dim, arch.b)
        self._forward_map = build_forward(self.obs_dim, arch.z_dim, self.action_dim, arch.f)
        self._actor = build_actor(self.obs_dim, arch.z_dim, self.action_dim, arch.actor)
        self._obs_normalizer = nn.BatchNorm1d(self.obs_dim, affine=False, momentum=0.01) if self.params.norm_obs else nn.Identity()
        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(torch_helpers.device)

    def _prepare_for_train(self) -> None:
        # create TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        # if device is not None:
        #     self.params.device = device.type  # type: ignore
        return super().to(*args, **kwargs)

    def _normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor):
        return self._backward_map(self._normalize(obs))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._forward_map(self._normalize(obs), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        return self._actor(self._normalize(obs), z, std)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.params.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.params.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.params.actor_std)
        if mean:
            return dist.mean
        return dist.sample()
    