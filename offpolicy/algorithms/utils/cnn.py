import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from offpolicy.utils.util import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size

        if self._use_orthogonal:
            def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(
                x, 0), nn.init.calculate_gain('relu'))
        else:
            def init_(m): return init(m, nn.init.xavier_uniform_, lambda x: nn.init.constant_(
                x, 0), gain=nn.init.calculate_gain('relu'))

        inputs_dim = obs_shape[0]
        image_size = obs_shape[1]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(inputs_dim, 32, 3, stride=1)), nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * (image_size-3+1) * \
                            (image_size-3+1), self.hidden_size)), nn.ReLU(),
            init_(nn.Linear(self.hidden_size, self.hidden_size)), nn.ReLU())

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)

        return x
