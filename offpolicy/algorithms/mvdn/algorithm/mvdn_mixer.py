import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from offpolicy.utils.util import to_torch


class M_VDNMixer(nn.Module):
    """
    Computes total Q values given agent q values and global states.
    :param args: (namespace) contains information about hyperparameters and algorithm configuration (unused).
    :param num_agents: (int) number of agents in env
    :param cent_obs_dim: (int) dimension of the centralized state (unused).
    :param device: (torch.Device) torch device on which to do computations.
    :param multidiscrete_list: (list) list of each action dimension if action space is multidiscrete
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

    def forward(self, agent_q_inps):
        """
        Computes Q_tot by summing individual agent q values.
        :param agent_q_inps: (torch.Tensor) individual agent q values

        :return Q_tot: (torch.Tensor) computed Q_tot values
        """
        agent_q_inps = to_torch(agent_q_inps)

        return agent_q_inps.sum(dim=-1).view(-1, 1, 1)
