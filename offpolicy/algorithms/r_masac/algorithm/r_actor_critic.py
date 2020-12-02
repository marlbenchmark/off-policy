import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, OneHotCategorical, Categorical
from offpolicy.utils.util import init, get_clones, gumbel_softmax, onehot_from_logits
from offpolicy.utils.mlp import MLPLayer

epsilon = 1e-6
# constants used in baselines implementation, might need to change
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class R_Critic(nn.Module):
    def __init__(self, args, central_obs_dim, central_act_dim, device, discrete):
        super(R_Critic, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self.device = device

        input_dim = central_obs_dim + central_act_dim

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
        self.norm = nn.LayerNorm(self.hidden_size).to(self.device)

        if self._use_orthogonal:
            def init_(m): return init(m, nn.init.orthogonal_,
                                      lambda x: nn.init.constant_(x, 0))
        else:
            def init_(m): return init(m, nn.init.xavier_uniform_,
                                      lambda x: nn.init.constant_(x, 0))

        self.q1_out = init_(nn.Linear(self.hidden_size, 1)).to(self.device)
        self.q2_out = init_(nn.Linear(self.hidden_size, 1)).to(self.device)

    def forward(self, central_obs, central_act, rnn_hidden_states):
        # ensure inputs are torch tensors
        if type(central_obs) == np.ndarray:
            central_obs = torch.FloatTensor(central_obs)
        if type(central_act) == np.ndarray:
            central_act = torch.FloatTensor(central_act)
        if type(rnn_hidden_states) == np.ndarray:
            rnn_hidden_states = torch.FloatTensor(rnn_hidden_states)

        no_sequence = False
        if len(central_obs.shape) == 2:
            # no sequence, so add a time dimension of len 0
            no_sequence = True
            central_obs = central_obs[None]

        if len(rnn_hidden_states.shape) == 2:
            # also add a first dimension to the rnn hidden states
            rnn_hidden_states = rnn_hidden_states[None]

        if len(central_act.shape) == 2:
            central_act = central_act[None]
        x = torch.cat([central_obs, central_act], dim=2)

        x = x.to(self.device)
        rnn_hidden_states = rnn_hidden_states.to(self.device)

        if self._use_feature_normalization:
            x = self.feature_norm(x)

        rnn_inp = self.mlp(x)
        self.rnn.flatten_parameters()
        rnn_outs, h_final = self.rnn(rnn_inp, rnn_hidden_states)
        rnn_outs = self.norm(rnn_outs)
        q1_values = self.q1_out(rnn_outs)
        q2_values = self.q2_out(rnn_outs)

        if no_sequence:
            # remove the time dimension
            q1_values = q1_values[0, :, :]
            q2_values = q2_values[0, :, :]

        h_final = h_final[0, :, :]

        q1_values = q1_values.cpu()
        q2_values = q2_values.cpu()
        h_final = h_final.cpu()

        return q1_values, q2_values, h_final


class R_DiscreteActor(nn.Module):
    def __init__(self, args, obs_dim, act_dim, device, take_prev_action=False):
        super(R_DiscreteActor, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = args.hidden_size
        self.device = device
        self.take_prev_act = take_prev_action

        if take_prev_action:
            input_dim = obs_dim + act_dim
        else:
            input_dim = obs_dim

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim).to(self.device)

        # map observation input into input for rnn
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
        self.norm = nn.LayerNorm(self.hidden_size).to(self.device)
        # get action from rnn hidden state
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

    def forward(self, obs, prev_acts, rnn_hidden_states):
        # make sure input is a torch tensor
        if type(obs) == np.ndarray:
            obs = torch.FloatTensor(obs)
        if prev_acts is not None and type(prev_acts) == np.ndarray:
            prev_acts = torch.FloatTensor(prev_acts)
        if type(rnn_hidden_states) == np.ndarray:
            rnn_hidden_states = torch.FloatTensor(rnn_hidden_states)

        no_sequence = False
        if len(obs.shape) == 2:
            # this means we're just getting one output (no sequence)
            no_sequence = True
            obs = obs[None]
            if self.take_prev_act:
                prev_acts = prev_acts[None]
            # x is now of shape (seq_len, batch_size, obs_dim)

        if self.take_prev_act:
            inp = torch.cat((obs, prev_acts.float()), dim=-1)
        else:
            inp = obs

        if len(rnn_hidden_states.shape) == 2:
            # hiddens should be of shape (1, batch_size, dim)
            rnn_hidden_states = rnn_hidden_states[None]

        inp = inp.to(self.device)
        rnn_hidden_states = rnn_hidden_states.to(self.device)

        if self._use_feature_normalization:
            inp = self.feature_norm(inp)
        # get RNN input
        rnn_inp = self.mlp(inp)
        # pass RNN input and hidden states through RNN to get the hidden state sequence and the final hidden
        self.rnn.flatten_parameters()
        rnn_outs, h_final = self.rnn(rnn_inp, rnn_hidden_states)
        rnn_outs = self.norm(rnn_outs)
        # pass outputs through linear layer # TODO: should i put a activation in between this??

        if self.multidiscrete:
            act_out_logits = []
            for a_out in self.action_outs:
                act_out_logit = a_out(rnn_outs)
                if no_sequence:
                    # remove the dummy first time dimension if the input didn't have a time dimension
                    act_out_logit = act_out_logit[0, :, :]
                act_out_logits.append(act_out_logit.cpu())
        else:
            act_out_logits = self.action_out(rnn_outs)
            if no_sequence:
                # remove the dummy first time dimension if the input didn't have a time dimension
                act_out_logits = act_out_logits[0, :, :]
            act_out_logits = act_out_logits.cpu()

        h_final = h_final.cpu()

        # remove the first hidden dimension before returning
        return act_out_logits, h_final[0, :, :]

    def sample(self, obs, prev_acts, rnn_hidden_states, available_actions=None, sample_gumbel=False):
        # TODO: review this method
        act_logits, h_outs = self.forward(obs, prev_acts, rnn_hidden_states)

        if self.multidiscrete:
            sampled_actions = []
            mean_action_logprobs = []
            max_prob_actions = []
            for act_logit in act_logits:
                categorical = OneHotCategorical(logits=act_logit)

                all_action_prob = categorical.probs
                eps = (all_action_prob == 0.0) * 1e-6
                all_action_logprob = torch.log(
                    all_action_prob + eps.float().detach())
                mean_action_logprob = (
                    all_action_logprob * all_action_prob).sum(dim=-1).unsqueeze(-1)

                if sample_gumbel:
                    # get a differentiable sample of the action
                    sampled_action = gumbel_softmax(act_logit, hard=True)
                else:
                    sampled_action = categorical.sample()

                max_prob_action = onehot_from_logits(act_logit)

                sampled_actions.append(sampled_action)
                mean_action_logprobs.append(mean_action_logprob)
                max_prob_actions.append(max_prob_action)

            sampled_actions = torch.cat(sampled_actions, dim=-1)
            mean_action_logprobs = torch.cat(mean_action_logprobs, dim=-1)
            max_prob_actions = torch.cat(max_prob_actions, dim=-1)

            return sampled_actions, mean_action_logprobs, max_prob_actions, h_outs
        else:
            categorical = OneHotCategorical(logits=act_logits)

            all_action_probs = categorical.probs
            eps = (all_action_probs == 0.0) * 1e-6
            all_action_logprobs = torch.log(
                all_action_probs + eps.float().detach())
            mean_action_logprobs = (
                all_action_logprobs * all_action_probs).sum(dim=-1).unsqueeze(-1)

            if sample_gumbel:
                # get a differentiable sample of the action
                sampled_actions = gumbel_softmax(
                    act_logits, available_actions, hard=True)
            else:
                if available_actions is not None:
                    if type(available_actions) == np.ndarray:
                        available_actions = torch.FloatTensor(
                            available_actions)
                    act_logits[available_actions == 0] = -1e10
                    sampled_actions = OneHotCategorical(
                        logits=act_logits).sample()
                else:
                    sampled_actions = categorical.sample()

            max_prob_actions = onehot_from_logits(
                act_logits, available_actions)
            return sampled_actions, mean_action_logprobs, max_prob_actions, h_outs


class R_GaussianActor(nn.Module):
    def __init__(self, args, obs_dim, act_dim, action_space, device, take_prev_action=False):
        super(R_GaussianActor, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = args.hidden_size
        self.device = device
        self.take_prev_act = take_prev_action

        if take_prev_action:
            input_dim = obs_dim + act_dim
        else:
            input_dim = obs_dim

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim).to(self.device)

        # map observation input into input for rnn
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
        self.norm = nn.LayerNorm(self.hidden_size).to(self.device)

        # get action from rnn hidden state
        if self._use_orthogonal:
            def init_(m): return init(m, nn.init.orthogonal_,
                                      lambda x: nn.init.constant_(x, 0))
        else:
            def init_(m): return init(m, nn.init.xavier_uniform_,
                                      lambda x: nn.init.constant_(x, 0))

        self.mean_layer = init_(
            nn.Linear(self.hidden_size, self.act_dim)).to(self.device)
        self.log_std_layer = init_(
            nn.Linear(self.hidden_size, self.act_dim)).to(self.device)

        # SAC rescaling to respect action bounds (see paper)
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.tensor(
                (action_space.high - action_space.low) / 2.).float()

            self.action_bias = torch.tensor(
                (action_space.high + action_space.low) / 2.).float()

    def init_hidden(self, batch_size):
        return self.mlp.fc1[0].weight.new_zeros(batch_size, self.hidden_size)

    def forward(self, obs, prev_acts, rnn_hidden_states):
        # make sure input is a torch tensor
        if type(obs) == np.ndarray:
            obs = torch.FloatTensor(obs)
        if prev_acts is not None and type(prev_acts) == np.ndarray:
            prev_acts = torch.FloatTensor(prev_acts)
        if type(rnn_hidden_states) == np.ndarray:
            rnn_hidden_states = torch.FloatTensor(rnn_hidden_states)

        no_sequence = False
        if len(obs.shape) == 2:
            # this means we're just getting one output (no sequence)
            no_sequence = True
            obs = obs[None]
            if self.take_prev_act:
                prev_acts = prev_acts[None]
            # x is now of shape (seq_len, batch_size, obs_dim)

        if self.take_prev_act:
            inp = torch.cat((obs, prev_acts), dim=-1)
        else:
            inp = obs

        if len(rnn_hidden_states.shape) == 2:
            # hiddens should be of shape (1, batch_size, dim)
            rnn_hidden_states = rnn_hidden_states[None]

        inp = inp.to(self.device)
        rnn_hidden_states = rnn_hidden_states.to(self.device)

        if self._use_feature_normalization:
            inp = self.feature_norm(inp)

        # get RNN input
        rnn_inp = self.mlp(inp)
        self.rnn.flatten_parameters()
        # pass RNN input and hidden states through RNN to get the hidden state sequence and the final hidden
        rnn_outs, h_final = self.rnn(rnn_inp, rnn_hidden_states)
        rnn_outs = self.norm(rnn_outs)
        # pass outputs through linear layer # TODO: should i put a activation in between this??
        mean_outs = self.mean_layer(rnn_outs)
        log_std_outs = self.log_std_layer(rnn_outs)

        if no_sequence:
            # remove the dummy first time dimension if the input didn't have a time dimension
            mean_outs = mean_outs[0, :, :]
            log_std_outs = log_std_outs[0, :, :]

        # remove the first hidden dimension before returning
        mean_outs = mean_outs.cpu()
        log_std_outs = log_std_outs.cpu()
        h_final = h_final.cpu()

        return mean_outs, log_std_outs, h_final[0, :, :]

    def sample(self, obs, prev_acts, rnn_hidden_states, available_actions=None):
        # TODO: review this method
        means, log_stds, h_outs = self.forward(
            obs, prev_acts, rnn_hidden_states)
        stds = log_stds.exp()
        normal = Normal(means, stds)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        sampled_actions = y_t * self.action_scale + self.action_bias
        log_probs = normal.log_prob(x_t)
        log_probs -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_probs = log_probs.sum(2, keepdim=True)
        means = torch.tanh(means) * self.action_scale + self.action_bias

        return sampled_actions, log_probs, means, h_outs
