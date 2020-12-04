import torch
import numpy as np
from torch.distributions import OneHotCategorical
from offpolicy.algorithms.maddpg.algorithm.actor_critic import Actor, Critic
from offpolicy.utils.util import get_state_dim, is_discrete, is_multidiscrete, get_dim_from_space, DecayThenFlatSchedule, soft_update, hard_update, \
    gumbel_softmax, onehot_from_logits, make_onehot, gaussian_noise, avail_choose


class MADDPGPolicy:
    def __init__(self, config, policy_config, train=True):

        self.config = config
        self.device = config['device']
        self.args = self.config["args"]
        self.tau = self.args.tau
        self.lr = self.args.lr
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay

        self.central_obs_dim, self.central_act_dim = policy_config[
            "cent_obs_dim"], policy_config["cent_act_dim"]
        self.obs_space = policy_config["obs_space"]
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.discrete = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)

        self.act_dim = get_dim_from_space(self.act_space)

        self.actor = Actor(self.args, self.obs_dim,
                           self.act_dim, self.device)
        self.critic = Critic(self.args, self.central_obs_dim,
                             self.central_act_dim, self.device)

        self.target_actor = Actor(
            self.args, self.obs_dim, self.act_dim, self.device)
        self.target_critic = Critic(
            self.args, self.central_obs_dim, self.central_act_dim, self.device)

        # sync the target weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        if train:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(
            ), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(
            ), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

            if self.discrete_action:
                # eps greedy exploration
                self.exploration = DecayThenFlatSchedule(self.args.epsilon_start, self.args.epsilon_finish,
                                                         self.args.epsilon_anneal_time, decay="linear")


    def get_actions(self, obs, available_actions=None, t_env=None, explore=False, use_target=False, use_gumbel=False):
        batch_size = obs.shape[0]
        eps = None
        if use_target:
            actor_out = self.target_actor(obs)
        else:
            actor_out = self.actor(obs)

        if self.discrete_action:
            if self.multidiscrete:
                if use_gumbel or explore:
                    onehot_actions = list(
                        map(lambda a: gumbel_softmax(a, hard=True), actor_out))
                else:
                    onehot_actions = list(map(onehot_from_logits, actor_out))

                onehot_actions = torch.cat(onehot_actions, dim=-1)
                if explore:
                    # eps greedy exploration
                    batch_size = obs.shape[0]
                    eps = self.exploration.eval(t_env)
                    rand_numbers = np.random.rand(batch_size, 1)
                    take_random = (rand_numbers < eps).astype(int).reshape(-1, 1)

                    # random actions sample uniformly from action space
                    random_actions = [OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i])).sample() for i in range(len(self.act_dim))]
                    random_actions = torch.cat(random_actions, dim=1)
                    actions = (1 - take_random) * onehot_actions.detach().cpu().numpy() + \
                        take_random * random_actions.detach().cpu().numpy()
                else:
                    actions = onehot_actions
            else:
                if use_gumbel or explore:
                    onehot_actions = gumbel_softmax(actor_out, available_actions, hard=True)  # gumbel has a gradient
                    actions = onehot_actions
                else:
                    onehot_actions = onehot_from_logits(actor_out, available_actions)  # no gradient

                if explore:
                    # eps greedy exploration
                    eps = self.exploration.eval(t_env)
                    rand_numbers = np.random.rand(batch_size, 1)
                    # random actions sample uniformly from action space
                    logits = torch.ones(batch_size, self.act_dim)
                    random_actions = avail_choose(logits, available_actions).sample()
                    random_actions = make_onehot(random_actions, self.act_dim)
                    take_random = (rand_numbers < eps).astype(int)
                    actions = (1 - take_random) * onehot_actions.detach().cpu().numpy() + take_random * random_actions
                else:
                    actions = onehot_actions
        else:
            if explore:
                actions = gaussian_noise(actor_out.shape, self.args.act_noise_std) + actor_out
            else:
                actions = actor_out
            # # clip the actions at the bounds of the action space
            # actions = torch.max(torch.min(actions, torch.from_numpy(self.act_space.high)), torch.from_numpy(self.act_space.low))

        return actions, eps

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

    def soft_target_updates(self):
        # polyak updates to target networks
        soft_update(self.target_critic, self.critic, self.args.tau)
        soft_update(self.target_actor, self.actor, self.args.tau)

    def hard_target_updates(self):
        # polyak updates to target networks
        hard_update(self.target_critic, self.critic)
        hard_update(self.target_actor, self.actor)
