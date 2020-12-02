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
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size).to(self.device)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)

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

    def init_hidden(self, batch_size):
        return self.q.weight.new_zeros(batch_size, self.hidden_size)

    def forward(self, x, rnn_hidden_states):

        if type(x) == np.ndarray:
            x = torch.FloatTensor(x)
        if type(rnn_hidden_states) == np.ndarray:
            rnn_hidden_states = torch.FloatTensor(rnn_hidden_states)

        no_sequence = False
        if len(x.shape) == 2:
            # x is just batch size x inp_dim, so make it to shape 1 x batch_size x inp_dim
            no_sequence = True
            x = x[None]

        if len(rnn_hidden_states.shape) == 2:
            rnn_hidden_states = rnn_hidden_states[None]

        x = x.to(self.device)
        rnn_hidden_states = rnn_hidden_states.to(self.device)

        if self._use_feature_normalization:
            x = self.feature_norm(x)

        rnn_inp = self.mlp(x)
        self.rnn.flatten_parameters()
        rnn_outs, h_final = self.rnn(rnn_inp, rnn_hidden_states)

        if self.multidiscrete:
            q_outs = []
            for q_out in self.q_outs:
                q_out = q_out(rnn_outs)
                if no_sequence:
                    # remove the dummy first time dimension if the input didn't have a time dimension
                    q_out = q_out[0, :, :]
                q_out = q_out.cpu()
                q_outs.append(q_out)
        else:
            q_outs = self.q_out(rnn_outs)
            if no_sequence:
                # remove the dummy first time dimension if the input didn't have a time dimension
                q_outs = q_outs[0, :, :]
            q_outs = q_outs
            q_outs = q_outs.cpu()

        h_final = h_final.cpu()

        return q_outs, h_final[0]
