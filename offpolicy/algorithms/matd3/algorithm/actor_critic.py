import copy
import numpy as np

import torch
import torch.nn as nn

from offpolicy.utils.util import init, get_clones
from offpolicy.utils.mlp import MLPLayer

class Actor(nn.Module):
    def __init__(self, args, obs_dim, act_dim, discrete_action, device):
        super(Actor, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = args.hidden_size
        self.discrete = discrete_action
        self.device = device

        input_dim = self.obs_dim

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim).to(self.device)
        self.mlp = MLPLayer(input_dim, self.hidden_size, self._layer_N,
                            self._use_orthogonal, self._use_ReLU).to(self.device)

        if self._use_orthogonal:
            def init_(m): return init(m, nn.init.orthogonal_,
                                      lambda x: nn.init.constant_(x, 0), self._gain)
        else:
            def init_(m): return init(m, nn.init.xavier_uniform_,
                                      lambda x: nn.init.constant_(x, 0), self._gain)

        if isinstance(act_dim, np.ndarray):
            # MultiDiscrete setting: have n Linear layers for each action
            self.multidiscrete = True
            self.action_outs = [init_(nn.Linear(self.hidden_size, a_dim)).to(
                self.device) for a_dim in act_dim]
        else:
            self.multidiscrete = False
            self.action_out = init_(
                nn.Linear(self.hidden_size, act_dim)).to(self.device)

    def forward(self, x):
        # make sure input is a torch tensor
        if type(x) == np.ndarray:
            x = torch.FloatTensor(x)

        x = x.to(self.device)

        # get RNN input
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        if self.multidiscrete:
            actions = [a_out(x).cpu() for a_out in self.action_outs]
        else:
            actions = self.action_out(x).cpu()

        return actions


class Critic(nn.Module):
    def __init__(self, args, central_obs_dim, central_act_dim, device):
        super(Critic, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self.central_obs_dim = central_obs_dim
        self.central_act_dim = central_act_dim
        self.hidden_size = args.hidden_size
        self.device = device

        input_dim = central_obs_dim + central_act_dim
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim).to(self.device)

        # map observation input into input for rnn
        self.mlp1 = MLPLayer(input_dim, self.hidden_size, self._layer_N,
                             self._use_orthogonal, self._use_ReLU).to(self.device)
        self.mlp2 = MLPLayer(input_dim, self.hidden_size, self._layer_N,
                             self._use_orthogonal, self._use_ReLU).to(self.device)

        if self._use_orthogonal:
            def init_(m): return init(m, nn.init.orthogonal_,
                                      lambda x: nn.init.constant_(x, 0))
        else:
            def init_(m): return init(m, nn.init.xavier_uniform_,
                                      lambda x: nn.init.constant_(x, 0))

        self.q1_out = init_(nn.Linear(self.hidden_size, 1)).to(self.device)
        self.q2_out = init_(nn.Linear(self.hidden_size, 1)).to(self.device)

    def forward(self, central_obs, central_act):
        if type(central_obs) == np.ndarray:
            central_obs = torch.FloatTensor(central_obs)
        if type(central_act) == np.ndarray:
            central_act = torch.FloatTensor(central_act)

        central_obs = central_obs.to(self.device)
        central_act = central_act.to(self.device)

        x = torch.cat([central_obs, central_act], dim=1)

        if self._use_feature_normalization:
            x = self.feature_norm(x)

        q1 = self.mlp1(x)
        q2 = self.mlp2(x)

        q1_value = self.q1_out(q1)
        q2_value = self.q2_out(q2)

        q1_value = q1_value.cpu()
        q2_value = q2_value.cpu()

        return q1_value, q2_value

    def Q1(self, central_obs, central_act):
        if type(central_obs) == np.ndarray:
            central_obs = torch.FloatTensor(central_obs)
        if type(central_act) == np.ndarray:
            central_act = torch.FloatTensor(central_act)

        central_obs = central_obs.to(self.device)
        central_act = central_act.to(self.device)

        x = torch.cat([central_obs, central_act], dim=1)

        if self._use_feature_normalization:
            x = self.feature_norm(x)

        q1 = self.mlp1(x)
        q1_value = self.q1_out(q1)

        q1_value = q1_value.cpu()

        return q1_value
