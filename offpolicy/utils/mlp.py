import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from offpolicy.utils.util import init, get_clones

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        if use_orthogonal:
            if use_ReLU:
                active_func = nn.ReLU()

                def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
                    x, 0), gain=nn.init.calculate_gain('relu'))
            else:
                active_func = nn.Tanh()

                def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
                    x, 0), gain=nn.init.calculate_gain('tanh'))
        else:
            if use_ReLU:
                active_func = nn.ReLU()

                def init_(m): return init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(
                    x, 0), gain=nn.init.calculate_gain('relu'))
            else:
                active_func = nn.Tanh()
                def init_(m): return init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(
                    x, 0), gain=nn.init.calculate_gain('tanh'))

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x
