import torch
import numpy as np
from torch.distributions import Normal, OneHotCategorical
from offpolicy.algorithms.r_masac.algorithm.r_actor_critic import R_GaussianActor, R_DiscreteActor, R_Critic
from offpolicy.utils.util import is_discrete, is_multidiscrete, get_dim_from_space, soft_update, hard_update, \
    gumbel_softmax, onehot_from_logits, avail_choose
from offpolicy.algorithms.common.recurrent_policy import RecurrentPolicy

class R_MASACPolicy(RecurrentPolicy):
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
        self.discrete = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)

        if self.discrete:
            self.actor = R_DiscreteActor(self.args, self.obs_dim, self.act_dim, self.device, take_prev_action=self.prev_act_inp)
            # slightly less than max possible entropy
            self.target_entropy = -np.log((1.0 / self.act_dim)) * self.target_entropy_coef # ! check this 
        else:
            self.actor = R_GaussianActor(self.args, self.obs_dim, self.act_dim, self.device, take_prev_action=self.prev_act_inp)
            # max possible entropy
            self.target_entropy = -torch.prod(torch.Tensor(self.act_space.shape)).item()
            # SAC rescaling to respect action bounds (see paper)
            self.action_scale = torch.tensor((self.act_space.high - self.act_space.low) / 2.).float()
            self.action_bias = torch.tensor((self.act_space.high + self.act_space.low) / 2.).float()

        self.critic = R_Critic(self.args, self.central_obs_dim, self.central_act_dim, self.device)
        self.target_critic = R_Critic(self.args, self.central_obs_dim, self.central_act_dim, self.device)
        # sync the target weights
        self.target_critic.load_state_dict(self.critic.state_dict())

        if train:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

            # will get updated via log_alpha
            self.alpha = self.config["args"].alpha
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def get_actions(self, obs, prev_actions, rnn_states, available_actions=None, t_env=None, explore=False, use_target=None, use_gumbel=False):

        if self.discrete:
            actions, h_outs, log_probs =self.get_actions_discrete(obs, prev_actions, rnn_states, available_actions, explore)
        else:
            actions, h_outs, log_probs = self.get_actions_continuous(obs, prev_actions, rnn_states, explore=explore)

        return actions, h_outs, log_probs
    
    def get_actions_continuous(self, obs, prev_actions, actor_rnn_states, available_actions=None, explore=False):
        means, log_stds, h_outs = self.actor(obs, prev_actions, actor_rnn_states)
        
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

    def get_actions_discrete(self, obs, prev_actions, actor_rnn_states, available_actions=None, explore=False, use_gumbel=False):
        act_logits, h_outs = self.actor(obs, prev_actions, actor_rnn_states)

        if self.multidiscrete:
            actions = []
            dist_entropies = []
            for act_logit in act_logits:
                categorical = OneHotCategorical(logits=act_logit)

                action_prob = categorical.probs
                eps = (action_prob == 0.0) * 1e-6
                action_logprob = torch.log(action_prob + eps.float().detach())
                dist_entropy = (action_logprob * action_prob).sum(dim=-1).unsqueeze(-1)

                if use_gumbel:
                    # get a differentiable sample of the action
                    action = gumbel_softmax(act_logit, hard=True)
                elif explore:
                    action = categorical.sample()
                else:
                    action = onehot_from_logits(act_logit)

                actions.append(action)
                dist_entropies.append(dist_entropy)

            actions = torch.cat(actions, dim=-1)
            dist_entropies = torch.cat(dist_entropies, dim=-1)
        else:
            categorical = OneHotCategorical(logits=act_logits)
            action_probs = categorical.probs
            eps = (action_probs == 0.0) * 1e-8
            action_logprobs = torch.log(action_probs + eps.float().detach())
            dist_entropies = (action_logprobs * action_probs).sum(dim=-1).unsqueeze(-1)

            if use_gumbel:
                # get a differentiable sample of the action               
                actions = gumbel_softmax(act_logits, available_actions, hard=True, device=self.device)
            elif explore:
                actions = OneHotCategorical(logits=avail_choose(act_logits, available_actions)).sample()               
            else:
                actions = onehot_from_logits(act_logits, available_actions)

        return actions, h_outs, dist_entropies

    def get_random_actions(self, obs, available_actions=None):
        batch_size = obs.shape[0]

        if self.discrete:
            if self.multidiscrete:
                random_actions = [OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i])).sample().numpy() for i in
                                    range(len(self.act_dim))]
                random_actions = np.concatenate(random_actions, axis=-1)
            else:
                if available_actions is not None:
                    logits = avail_choose(torch.ones(batch_size, self.act_dim), available_actions)
                    random_actions = OneHotCategorical(logits=logits).sample().numpy()
                else:
                    random_actions = OneHotCategorical(logits=torch.ones(batch_size, self.act_dim)).sample().numpy()
        else:
            random_actions = np.random.uniform(self.act_space.low, self.act_space.high, size=(batch_size, self.act_dim))

        return random_actions

    def init_hidden(self, num_agents, batch_size):
        if num_agents == -1:
            return torch.zeros(batch_size, self.hidden_size)
        else:
            return torch.zeros(num_agents, batch_size, self.hidden_size)

    def soft_target_updates(self):
        # polyak updates to target networks
        soft_update(self.target_critic, self.critic, self.tau)

    def hard_target_updates(self):
        # polyak updates to target networks
        hard_update(self.target_critic, self.critic)
