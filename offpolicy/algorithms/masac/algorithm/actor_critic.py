import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, OneHotCategorical, Categorical
from offpolicy.utils.util import init, get_clones, gumbel_softmax, onehot_from_logits
from offpolicy.utils.mlp import MLPLayer

# constants used in baselines implementation, might need to change
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class Critic(nn.Module):
    def __init__(self, args, central_obs_dim, central_act_dim, device):
        super(Critic, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.device = device

        input_dim = central_obs_dim + central_act_dim

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim).to(self.device)

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


class DiscreteActor(nn.Module):

    def __init__(self, args, obs_dim, act_dim, device):
        super(DiscreteActor, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.device = device

        input_dim = self.obs_dim

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim).to(self.device)

        # map observation input into input for rnn
        self.mlp = MLPLayer(input_dim, self.hidden_size, self._layer_N,
                            self._use_orthogonal, self._use_ReLU).to(self.device)

        # get action from rnn hidden state
        if self._use_orthogonal:
            def init_(m): return init(m, nn.init.orthogonal_,
                                      lambda x: nn.init.constant_(x, 0), self._gain)
        else:
            def init_(m): return init(m, nn.init.xavier_uniform_,
                                      lambda x: nn.init.constant_(x, 0), self._gain)

        if isinstance(self.act_dim, np.ndarray):
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

        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        if self.multidiscrete:
            act_out_logits = []
            for a_out in self.action_outs:
                act_out_logit = a_out(x)
                act_out_logits.append(act_out_logit.cpu())
        else:
            act_out_logits = self.action_out(x)
            act_out_logits = act_out_logits.cpu()

        return act_out_logits

    def sample(self, x, available_actions=None, sample_gumbel=False):
        # TODO: review this method
        act_logits = self.forward(x)

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

            return sampled_actions, mean_action_logprobs, max_prob_actions
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
            return sampled_actions, mean_action_logprobs, max_prob_actions


class GaussianActor(nn.Module):
    def __init__(self, args, obs_dim, act_dim, action_space, device):
        super(GaussianActor, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = args.hidden_size
        self.device = device

        input_dim = self.obs_dim

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim).to(self.device)

        # map observation input into input for rnn
        self.mlp = MLPLayer(input_dim, self.hidden_size, self._layer_N,
                            self._use_orthogonal, self._use_ReLU).to(self.device)

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

    def forward(self, x):

        x = x.to(self.device)

        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        mean = mean.cpu()
        log_std = log_std.cpu()

        return mean, log_std

    def sample(self, x, available_actions=None):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
