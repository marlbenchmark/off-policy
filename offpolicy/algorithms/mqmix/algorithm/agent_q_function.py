import torch
import torch.nn as nn

from offpolicy.utils.util import to_torch
from offpolicy.algorithms.utils.mlp import MLPBase
from offpolicy.algorithms.utils.act import ACTLayer

class AgentQFunction(nn.Module):
    # MLP implementation of the Agent Q function

    def __init__(self, args, input_dim, act_dim, device):
        # input dim is agent obs dim + agent acf dim
        # output dim is act dim
        super(AgentQFunction, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.mlp = MLPBase(args, input_dim)

        self.q = ACTLayer(act_dim, self.hidden_size, self._use_orthogonal, gain=self._gain)

        self.to(device)

    def forward(self, x):
        # make sure input is a torch tensor
        x = to_torch(x).to(**self.tpdv)
        x = self.mlp(x)
        # pass outputs through linear layer
        q_value = self.q(x)

        return q_value
