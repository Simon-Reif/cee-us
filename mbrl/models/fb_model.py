#adapts https://github.com/facebookresearch/metamotivo
from torch import nn

from mbrl import torch_helpers
from mbrl.models.fb_nn import build_actor, build_backward, build_forward


class FBModel(nn.Module):
    def __init__(self, params, obs_dim, action_dim, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        arch = params.archi
        # create networks
        self._backward_map = build_backward(obs_dim, arch.z_dim, arch.b)
        self._forward_map = build_forward(obs_dim, arch.z_dim, action_dim, arch.f)
        self._actor = build_actor(obs_dim, arch.z_dim, action_dim, arch.actor)
        self._obs_normalizer = nn.BatchNorm1d(obs_dim, affine=False, momentum=0.01) if self.cfg.norm_obs else nn.Identity()
        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(torch_helpers.device)


    