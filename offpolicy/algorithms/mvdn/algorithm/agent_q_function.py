import copy
import numpy as np

import torch
import torch.nn as nn

from offpolicy.utils.util import init, get_clones
from offpolicy.utils.mlp import MLPLayer

class AgentQFunction(nn.Module):
    # GRU implementation of the Agent Q function

    def __init__(self, input_dim, act_dim, args, device):
        # input dim is agent obs dim + agent acf dim
        # output dim is act dim
        super(AgentQFunction, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._layer_N = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.device = device

        # maps input to RNN input dimension
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim).to(self.device)
        self.mlp = MLPLayer(input_dim, self.hidden_size, self._layer_N,
                            self._use_orthogonal, self._use_ReLU).to(self.device)

        # get action from rnn hidden state
        if self._use_orthogonal:
            def init_(m): return init(m, nn.init.orthogonal_,
                                      lambda x: nn.init.constant_(x, 0), self._gain)
        else:
            def init_(m): return init(m, nn.init.xavier_uniform_,
                                      lambda x: nn.init.constant_(x, 0), self._gain)

        if isinstance(act_dim, np.ndarray):
            # MultiDiscrete setting: have n Linear layers for each q
            self.multidiscrete = True
            self.q_outs = [init_(nn.Linear(self.hidden_size, a_dim)).to(
                self.device) for a_dim in act_dim]
        else:
            self.multidiscrete = False
            self.q_out = init_(
                nn.Linear(self.hidden_size, act_dim)).to(self.device)

    def forward(self, x):

        if type(x) == np.ndarray:
            x = torch.FloatTensor(x)

        x = x.to(self.device)

        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        if self.multidiscrete:
            q_outs = [q_out(x).cpu() for q_out in self.q_outs]
        else:
            q_outs = self.q_out(x).cpu()

        return q_outs
