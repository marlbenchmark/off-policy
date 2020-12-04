import torch
import numpy as np
from torch.distributions import Normal
from offpolicy.algorithms.r_masac.algorithm.r_actor_critic import R_Actor, R_Critic
from offpolicy.utils.util import get_dim_from_space, soft_update, hard_update

class R_MASACPolicy:
    def __init__(self, config, policy_config, train=True):

        self.config = config
        self.device = config['device']
        self.args = self.config["args"]
        self.tau = self.args.tau
        self.lr = self.args.lr
        self.target_entropy_coef = self.args.target_entropy_coef
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay
        self.prev_act_inp = self.args.prev_act_inp

        self.central_obs_dim, self.central_act_dim = policy_config["cent_obs_dim"], policy_config["cent_act_dim"]
        self.obs_space = policy_config["obs_space"]
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.act_dim = get_dim_from_space(self.act_space)
        self.hidden_size = self.args.hidden_size

        self.actor = R_Actor(self.args, self.obs_dim, self.act_dim, self.device, take_prev_action=self.prev_act_inp)
        # max possible entropy
        self.target_entropy = -torch.prod(torch.Tensor(self.act_space.shape)).item()
        # SAC rescaling to respect action bounds (see paper)
        self.action_scale = torch.tensor((self.act_space,.high - self.act_space,.low) / 2.).float()
        self.action_bias = torch.tensor((self.act_space,.high + self.act_space,.low) / 2.).float()

        self.critic = R_Critic(self.args, self.central_obs_dim, self.central_act_dim, self.device, discrete=False)
        self.target_critic = R_Critic(self.args, self.central_obs_dim, self.central_act_dim, self.device, discrete=False)
        # sync the target weights
        self.target_critic.load_state_dict(self.critic.state_dict())

        if train:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

            # will get updated via log_alpha
            self.alpha = self.config["args"].alpha
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True).to(self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def get_actions(self, obs, prev_acts, rnn_states, avail_action=None, t_env=None, explore=False):
        # TODO: review this method
        means, log_stds, h_outs = self.actor(obs, prev_acts, rnn_states)
        
        stds = log_stds.exp()
        normal = Normal(means, stds)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        if explore:
            actions = y_t * self.action_scale + self.action_bias
        else:
            actions = torch.tanh(means) * self.action_scale + self.action_bias

        log_probs = normal.log_prob(x_t)
        log_probs -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_probs = log_probs.sum(2, keepdim=True)
        
        return actions, log_probs, h_outs

    def init_hidden(self, num_agents, batch_size, use_numpy=False):
        if use_numpy:
            if num_agents == -1:
                return np.zeros((batch_size, self.hidden_size), dtype=np.float32)
            else:
                return np.zeros((num_agents, batch_size, self.hidden_size), dtype=np.float32)
        else:
            if num_agents == -1:
                return torch.zeros(batch_size, self.hidden_size)
            else:
                return torch.zeros(num_agents, batch_size, self.hidden_size)

    def get_random_actions(self, obs, available_actions=None):
        batch_size = obs.shape[0]

        random_actions = np.random.uniform(self.act_space.low, self.act_space.high, size=(batch_size, self.act_dim))

        return random_actions

    def soft_target_updates(self):
        # polyak updates to target networks
        soft_update(self.target_critic, self.critic, self.tau)

    def hard_target_updates(self):
        # polyak updates to target networks
        hard_update(self.target_critic, self.critic)
