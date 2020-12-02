import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from offpolicy.utils.util import init


class M_VDNMixer(nn.Module):
    """
    computes Q_tot from individual Q_a values and the state
    """

    def __init__(self, args, num_agents, cent_obs_dim, device, multidiscrete_list=None):
        """
        init mixer class
        """
        super(M_VDNMixer, self).__init__()
        self.device = device
        self.num_agents = num_agents

        if multidiscrete_list:
            self.num_mixer_q_inps = sum(multidiscrete_list)
        else:
            self.num_mixer_q_inps = self.num_agents

    def forward(self, agent_q_inps, states):
        """outputs Q_tot, using the individual agent Q values and the centralized env state as inputs"""

        if type(agent_q_inps) == np.ndarray:
            agent_q_inps = torch.FloatTensor(agent_q_inps)
        if type(states) == np.ndarray:
            states = torch.FloatTensor(states)

        return agent_q_inps.sum(dim=-1).view(-1, 1, 1), None
