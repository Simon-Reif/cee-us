import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# code adapted from FB Paper
# maybe replace with more complex models

# Obs -> Embedding
class BackwardMap(nn.Module):
    def __init__(self, obs_dim, embed_dim):
        super(BackwardMap, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.backward_out = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        backward_value = self.backward_out(x)
        return backward_value

# Obs x Action x Embedding -> Embedding
class ForwardMap(nn.Module):
    def __init__(self, obs_dim, action_dim, embed_dim):
        super(ForwardMap, self).__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(obs_dim + embed_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.forward_out = nn.Linear(256, embed_dim)

    def forward(self, obs, w):
        w = w / torch.sqrt(1 + torch.norm(w, dim=-1, keepdim=True) ** 2 / self.embed_dim)
        x = torch.cat([obs, w], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        forward_value = self.forward_out(x)

        return forward_value.reshape(-1, self.embed_dim)