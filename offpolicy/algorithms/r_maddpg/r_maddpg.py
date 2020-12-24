import torch
import numpy as np
import copy
import itertools
from offpolicy.utils.util import huber_loss, mse_loss, check
from offpolicy.utils.popart import PopArt

class R_MADDPG:
    def __init__(self, args, num_agents, policies, policy_mapping_fn, device=None, episode_length=None):
        """Contains all policies and does policy updates"""
        self.args = args
        self.use_popart = self.args.use_popart
        self.use_value_active_masks = self.args.use_value_active_masks
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta
        self.tpdv = dict(dtype=torch.float32, device=device)

        if episode_length is None:
            self.episode_length = self.args.episode_length
        else:
            self.episode_length = episode_length

        self.num_agents = num_agents
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_ids = sorted(list(self.policies.keys()))
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in range(self.num_agents) if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}
        if self.use_popart:
            self.value_normalizer = {policy_id: PopArt(1) for policy_id in self.policies.keys()}

    def get_update_info(self, update_policy_id, obs_batch, act_batch, nobs_batch, avail_act_batch, navail_act_batch):
        """Get centralized state info for current timestep and next timestep, and also step forward target actor RNN"""
        act_sequences = []
        nact_sequences = []
        act_sequence_replace_ind_start = None
        # iterate through policies to get the target acts and other centralized info
        ind = 0
        for p_id in self.policy_ids:
            policy = self.policies[p_id]
            if p_id == update_policy_id:
                # where to start replacing actor actions from during actor update
                act_sequence_replace_ind_start = ind
            num_pol_agents = len(self.policy_agents[p_id])
            act_sequences.append(list(act_batch[p_id]))
            # get first observation for all agents under policy and stack them along batch dim
            first_obs = np.concatenate(obs_batch[p_id][:, 0], axis=0)
            # same with available actions
            if avail_act_batch[p_id] is not None:
                first_avail_act = np.concatenate(avail_act_batch[p_id][:, 0])
            else:
                first_avail_act = None
            total_batch_size = first_obs.shape[0]
            batch_size = total_batch_size // num_pol_agents
            # no gradient tracking is necessary for target actions
            with torch.no_grad():
                # step target actor through the first actions
                if isinstance(policy.act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(policy.act_dim))
                else:
                    sum_act_dim = policy.act_dim
                _, new_target_rnns, _ = policy.get_actions(first_obs,
                                                           np.zeros((total_batch_size, sum_act_dim), dtype=np.float32),
                                                           policy.init_hidden(-1, total_batch_size),
                                                           available_actions=first_avail_act,
                                                           use_target=True)
                # stack the nobs and acts of all the agents along batch dimension (data from buffer)
                combined_nobs_seq_batch = np.concatenate(nobs_batch[p_id], axis=1)
                combined_act_seq_batch = np.concatenate(act_batch[p_id], axis=1)
                if navail_act_batch[p_id] is not None:
                    combined_navail_act_batch = np.concatenate(navail_act_batch[p_id], axis=1)
                else:
                    combined_navail_act_batch = None
                # pass the entire buffer sequence of all agents to the target actor to get targ actions at each step
                pol_nact_seq, _, _ = policy.get_actions(combined_nobs_seq_batch,
                                                        combined_act_seq_batch,
                                                        new_target_rnns,
                                                        available_actions=combined_navail_act_batch,
                                                        use_target=True)
                # separate the actions into individual agent actions
                ind_agent_nact_seqs = pol_nact_seq.cpu().split(split_size=batch_size, dim=1)
            # cat to form centralized next step action
            nact_sequences.append(torch.cat(ind_agent_nact_seqs, dim=-1))
            # increase ind by number agents just processed
            ind += num_pol_agents
        # form centralized observations and actions by concatenating
        # flatten list of lists
        act_sequences = list(itertools.chain.from_iterable(act_sequences))
        cent_act_sequence_critic = np.concatenate(act_sequences, axis=-1)
        cent_nact_sequence = np.concatenate(nact_sequences, axis=-1)

        return cent_act_sequence_critic, act_sequences, act_sequence_replace_ind_start, cent_nact_sequence

    # @profile
    def shared_train_policy_on_batch(self, update_policy_id, batch, update_actor=None):
        # unpack the batch
        obs_batch, cent_obs_batch, \
        act_batch, rew_batch, \
        nobs_batch, cent_nobs_batch, \
        dones_batch, dones_env_batch, \
        avail_act_batch, navail_act_batch, \
        importance_weights, idxes = batch

        # obs_batch: dict mapping policy id to batches where each batch is shape (# agents, ep_len, batch_size, obs_dim)
        update_policy = self.policies[update_policy_id]
        # [episode length, episodes, agents, dim]
        batch_size = obs_batch[update_policy_id].shape[2]

        rew_sequence = check(rew_batch[update_policy_id][0]).to(**self.tpdv)
        # use numpy
        env_done_sequence = check(dones_env_batch[update_policy_id]).to(**self.tpdv)
        # mask the Q and target Q sequences with shifted dones (assume the first obs in episode is valid)
        first_step_dones = torch.zeros((1, env_done_sequence.shape[1], env_done_sequence.shape[2])).to(**self.tpdv)
        next_steps_dones = env_done_sequence[: self.episode_length - 1, :, :]
        curr_env_dones = torch.cat((first_step_dones, next_steps_dones), dim=0)

        cent_obs_sequence = cent_obs_batch[update_policy_id]
        cent_nobs_sequence = cent_nobs_batch[update_policy_id]
        # get centralized sequence information
        cent_act_sequence_buffer, act_sequences, act_sequence_replace_ind_start, cent_nact_sequence = \
            self.get_update_info(update_policy_id, obs_batch, act_batch,
                                 nobs_batch, avail_act_batch, navail_act_batch)
        

        # get sequence of Q value predictions: this will be a tensor of shape (ep len, batch size, 1)
        predicted_Q_sequence, _ = update_policy.critic(cent_obs_sequence, cent_act_sequence_buffer,
                                                       update_policy.init_hidden(-1, batch_size))

        # iterate over time to get target Qs since the history at each step should be formed from the buffer sequence
        next_Q_sequence = []
        # detach gradients since no gradients go through target critic
        with torch.no_grad():
            target_critic_rnn_state = update_policy.init_hidden(-1, batch_size)
            for t in range(self.episode_length):
                # update the RNN states based on the buffer sequence
                _, target_critic_rnn_state = update_policy.target_critic(cent_obs_sequence[t],
                                                                         cent_act_sequence_buffer[t],
                                                                         target_critic_rnn_state)
                # get the Q value using the next action taken by the target actor, but don't store the RNN state
                next_Q_t, _ = update_policy.target_critic(cent_nobs_sequence[t], 
                                                        cent_nact_sequence[t],
                                                        target_critic_rnn_state)
                next_Q_sequence.append(next_Q_t)

        # stack over time
        next_Q_sequence = torch.stack(next_Q_sequence)

        # mask the next step Qs and form targets; use the env dones as the mask since reward can accumulate even after 1 agent dies
        next_Q_sequence = (1 - env_done_sequence) * next_Q_sequence

        if self.use_popart:
            target_Q_sequence = rew_sequence + self.args.gamma * self.value_normalizer[update_policy_id].denormalize(next_Q_sequence)
            nodones_target_Q_sequence = target_Q_sequence[curr_env_dones == 0]
            target_Q_sequence[curr_env_dones == 0] = self.value_normalizer[update_policy_id](nodones_target_Q_sequence)
        else:
            target_Q_sequence = rew_sequence + self.args.gamma * next_Q_sequence

        predicted_Q_sequence = predicted_Q_sequence * (1 - curr_env_dones)
        target_Q_sequence = target_Q_sequence * (1 - curr_env_dones)
        # make sure to detach the targets! Loss is MSE loss, but divide by the number of unmasked elements
        # Mean bellman error for each timestep
        error = predicted_Q_sequence - target_Q_sequence.detach()
        if self.use_per:
            importance_weights = check(importance_weights).to(**self.tpdv)
            # prioritized experience replay
            if self.use_huber_loss:
                per_batch_critic_loss = huber_loss(error, self.huber_delta).sum(dim=0).flatten()
            else:
                per_batch_critic_loss = mse_loss(error).sum(dim=0).flatten()
            # weight each loss element by their importance sample weight
            importance_weight_critic_loss = per_batch_critic_loss * importance_weights
            critic_loss = importance_weight_critic_loss.sum() / (1 - curr_env_dones).sum()
            # new priorities are a combination of the maximum TD error across sequence and the mean TD error across sequence
            td_errors = error.abs().cpu().detach().numpy()
            new_priorities = ((1 - self.args.per_nu) * td_errors.mean(axis=0) +
                              self.args.per_nu * td_errors.max(axis=0)).flatten() + self.per_eps
        else:
            if self.use_huber_loss:
                critic_loss = huber_loss(error, self.huber_delta).sum() / (1 - curr_env_dones).sum()
            else:
                critic_loss = mse_loss(error).sum() / (1 - curr_env_dones).sum()
            new_priorities = None

        update_policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.critic.parameters(),
                                                                 self.args.max_grad_norm)
        update_policy.critic_optimizer.step()

        # actor update: can form losses for each agent that the update policy controls
        # freeze Q-networks
        for p in update_policy.critic.parameters():
            p.requires_grad = False

        agent_Q_sequences = []
        num_update_agents = len(self.policy_agents[update_policy_id])
        # formulate mask to determine how to combine actor output actions with batch output actions
        mask_temp = []
        for p_id in self.policy_ids:
            if isinstance(self.policies[p_id].act_dim, np.ndarray):
                # multidiscrete case
                sum_act_dim = int(sum(self.policies[p_id].act_dim))
            else:
                sum_act_dim = self.policies[p_id].act_dim
            for a_id in self.policy_agents[p_id]:
                mask_temp.append(np.zeros(sum_act_dim, dtype=np.float32))

        masks = []
        done_mask = []
        # need to iterate through agents, but only formulate masks at each step
        for i in range(num_update_agents):
            curr_mask_temp = copy.deepcopy(mask_temp)
            # set the mask to 1 at locations where the action should come from the actor output
            if isinstance(update_policy.act_dim, np.ndarray):
                # multidiscrete case
                sum_act_dim = int(sum(update_policy.act_dim))
            else:
                sum_act_dim = update_policy.act_dim
            curr_mask_temp[act_sequence_replace_ind_start + i] = np.ones(sum_act_dim, dtype=np.float32)
            curr_mask_vec = np.concatenate(curr_mask_temp)
            # expand this mask into the proper size
            curr_mask = np.tile(curr_mask_vec, (batch_size, 1))
            masks.append(curr_mask)

            # now collect agent dones
            # ! use numpy
            agent_done_sequence = check(dones_batch[update_policy_id][i])
            agent_first_step_dones = torch.zeros((1, agent_done_sequence.shape[1], agent_done_sequence.shape[2]))
            agent_next_steps_dones = agent_done_sequence[: self.episode_length - 1, :, :]
            curr_agent_dones = torch.cat((agent_first_step_dones, agent_next_steps_dones), dim=0)
            done_mask.append(curr_agent_dones)
        # cat masks and form into torch tensors
        mask = check(np.concatenate(masks)).to(**self.tpdv)
        done_mask = torch.cat(done_mask, dim=1).to(**self.tpdv)

        total_batch_size = batch_size * num_update_agents
        # stack obs, acts, and available acts of all agents along batch dimension to process at once
        pol_prev_buffer_act_seq = np.concatenate((np.zeros((1, total_batch_size, sum_act_dim), dtype=np.float32),
                                                  np.concatenate(act_batch[update_policy_id][:, : -1], axis=1)))
        pol_agents_obs_seq = np.concatenate(obs_batch[update_policy_id], axis=1)
        if avail_act_batch[update_policy_id] is not None:
            pol_agents_avail_act_seq = np.concatenate(avail_act_batch[update_policy_id], axis=1)
        else:
            pol_agents_avail_act_seq = None
        # get all the actions from actor, with gumbel softmax to differentiate through the samples
        policy_act_seq, _, _ = update_policy.get_actions(pol_agents_obs_seq, pol_prev_buffer_act_seq,
                                                         update_policy.init_hidden(-1, total_batch_size),
                                                         available_actions=pol_agents_avail_act_seq,
                                                         use_gumbel=True)
        
        # separate the output into individual agent act sequences
        agent_actor_seqs = policy_act_seq.split(split_size=batch_size, dim=1)
        # convert act sequences to torch, formulate centralized buffer action, and repeat as done above
        act_sequences = list(map(lambda arr: check(arr).to(**self.tpdv), act_sequences))

        actor_cent_acts = copy.deepcopy(act_sequences)
        for i in range(num_update_agents):
            actor_cent_acts[act_sequence_replace_ind_start + i] = agent_actor_seqs[i]
        # cat these along final dim to formulate centralized action and stack copies of the batch so all agents can be updated
        actor_cent_acts = torch.cat(actor_cent_acts, dim=-1).repeat((1, num_update_agents, 1))

        batch_cent_acts = torch.cat(act_sequences, dim=-1).repeat((1, num_update_agents, 1))
        # also repeat the cent obs
        stacked_cent_obs_seq = np.tile(cent_obs_sequence, (1, num_update_agents, 1))
        critic_rnn_state = update_policy.init_hidden(-1, total_batch_size)

        # iterate through timesteps and and get Q values to form actor loss
        for t in range(self.episode_length):
            # get Q values at timestep t with the replaced actions
            replaced_cent_act_batch = mask * actor_cent_acts[t] + (1 - mask) * batch_cent_acts[t]
            # get Q values at timestep but don't store the new RNN state
            Q_t, _ = update_policy.critic(stacked_cent_obs_seq[t], replaced_cent_act_batch, critic_rnn_state)
            # update the RNN state by stepping the RNN through with buffer sequence
            _, critic_rnn_state = update_policy.critic(stacked_cent_obs_seq[t], batch_cent_acts[t], critic_rnn_state)
            agent_Q_sequences.append(Q_t)
        # stack over time
        agent_Q_sequences = torch.stack(agent_Q_sequences)
        # mask at the places where agents were terminated in env
        agent_Q_sequences = agent_Q_sequences * (1 - done_mask)

        actor_loss = (-agent_Q_sequences).sum() / (1 - done_mask).sum()

        update_policy.critic_optimizer.zero_grad()
        update_policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.actor.parameters(),
                                                                self.args.max_grad_norm)
        update_policy.actor_optimizer.step()

        # unfreeze the Q networks
        for p in update_policy.critic.parameters():
            p.requires_grad = True

        train_info = {}
        
        train_info['critic_loss'] = critic_loss
        train_info['actor_loss'] = actor_loss
        train_info['critic_grad_norm'] = critic_grad_norm
        train_info['actor_grad_norm'] = actor_grad_norm

        return train_info, new_priorities, idxes

    # @profile
    def cent_train_policy_on_batch(self, update_policy_id, batch, update_actor=None):
        # unpack the batch
        obs_batch, cent_obs_batch, \
            act_batch, rew_batch, \
            nobs_batch, cent_nobs_batch, \
            dones_batch, dones_env_batch, \
            avail_act_batch, navail_act_batch, \
            importance_weights, idxes = batch
        # obs_batch: dict mapping policy id to batches where each batch is shape (# agents, ep_len, batch_size, obs_dim)
        update_policy = self.policies[update_policy_id]
        batch_size = obs_batch[update_policy_id].shape[2]

        rew_sequence = check(rew_batch[update_policy_id][0]).to(**self.tpdv)
        env_done_sequence = check(dones_env_batch[update_policy_id]).to(**self.tpdv)
        cent_obs_sequence = cent_obs_batch[update_policy_id]
        cent_nobs_sequence = cent_nobs_batch[update_policy_id]
        dones_sequence = dones_batch[update_policy_id]
        # get centralized sequence information
        cent_act_sequence_buffer, act_sequences, act_sequence_replace_ind_start, cent_nact_sequence = \
            self.get_update_info(update_policy_id, obs_batch, act_batch,
                                 nobs_batch, avail_act_batch, navail_act_batch)

        # combine all agents data into one array/tensor by stacking along batch dim; easier to process
        num_update_agents = len(self.policy_agents[update_policy_id])
        total_batch_size = batch_size * num_update_agents
        all_agent_cent_obs = np.concatenate(cent_obs_sequence, axis=1)
        all_agent_cent_nobs = np.concatenate(cent_nobs_sequence, axis=1)
        all_agent_dones = np.concatenate(dones_sequence, axis=1)
        # since this is same for each agent, just repeat when stacking
        all_agent_cent_act_buffer = np.tile(cent_act_sequence_buffer, (1, num_update_agents, 1))
        all_agent_cent_nact = np.tile(cent_nact_sequence, (1, num_update_agents, 1))
        all_env_dones = env_done_sequence.repeat(1, num_update_agents, 1)
        all_agent_rewards = rew_sequence.repeat(1, num_update_agents, 1)
        first_step_dones = torch.zeros((1, all_env_dones.shape[1], all_env_dones.shape[2])).to(**self.tpdv)
        next_steps_dones = all_env_dones[:-1, :, :]
        curr_env_dones = torch.cat((first_step_dones, next_steps_dones), dim=0)

        predicted_Q_sequence, _ = update_policy.critic(all_agent_cent_obs, all_agent_cent_act_buffer,
                                                       update_policy.init_hidden(-1, total_batch_size))
        # iterate over time to get target Qs since history at each step should be formed from the buffer sequence
        next_Q_sequence = []
        # don't track gradients for target computation
        with torch.no_grad():
            target_critic_rnn_state = update_policy.init_hidden(-1, total_batch_size)
            for t in range(self.episode_length):
                # update the RNN states based on the buffer sequence
                _, target_critic_rnn_state = update_policy.target_critic(all_agent_cent_obs[t],
                                                                         all_agent_cent_act_buffer[t],
                                                                         target_critic_rnn_state)
                # get the next value using next action taken by the target actor, but don't store the RNN state
                next_Q_t, _ = update_policy.target_critic(all_agent_cent_nobs[t], 
                                                        all_agent_cent_nact[t],
                                                        target_critic_rnn_state)
                next_Q_sequence.append(next_Q_t)
        # stack over time
        next_Q_sequence = torch.stack(next_Q_sequence)
        next_Q_sequence = (1 - all_env_dones) * next_Q_sequence
        if self.use_popart:
            
            target_Q_sequence = all_agent_rewards + self.args.gamma * \
                self.value_normalizer[update_policy_id].denormalize(next_Q_sequence)
            nodones_target_Q_sequence = target_Q_sequence[curr_env_dones == 0]
            target_Q_sequence[curr_env_dones == 0] = self.value_normalizer[update_policy_id](nodones_target_Q_sequence)
        else:
            target_Q_sequence = all_agent_rewards + self.args.gamma * next_Q_sequence

        predicted_Q_sequence = predicted_Q_sequence * (1 - curr_env_dones)
        target_Q_sequence = target_Q_sequence * (1 - curr_env_dones)

        if self.use_value_active_masks:
            curr_agent_dones = check(all_agent_dones).to(**self.tpdv)
            predicted_Q_sequence = predicted_Q_sequence * (1 - curr_agent_dones)
            target_Q_sequence = target_Q_sequence * (1 - curr_agent_dones)

        # make sure to detach the targets! Loss is MSE loss, but divide by the number of unmasked elements
        # Mean bellman error for each timestep
        error = predicted_Q_sequence - target_Q_sequence.detach()
        if self.use_per:
            agent_importance_weights = np.tile(importance_weights, num_update_agents)
            agent_importance_weights = check(agent_importance_weights).to(**self.tpdv)
            if self.use_huber_loss:
                per_batch_critic_loss = huber_loss(error, self.huber_delta).sum(dim=0).flatten()
            else:
                per_batch_critic_loss = mse_loss(error).sum(dim=0).flatten()
            agent_importance_weight_critic_loss = per_batch_critic_loss * agent_importance_weights

            if self.use_value_active_masks:
                critic_loss = agent_importance_weight_critic_loss.sum() / ((1 - curr_env_dones) * (1 - curr_agent_dones)).sum()
            else:
                critic_loss = agent_importance_weight_critic_loss.sum() / (1 - curr_env_dones).sum()

            td_errors = error.abs().cpu().detach().numpy()
            agent_new_priorities = ((1 - self.args.per_nu) * td_errors.mean(axis=0) + self.args.per_nu * td_errors.max(axis=0)).flatten()
            new_priorities = np.mean(np.split(agent_new_priorities, num_update_agents), axis=0) + self.per_eps
        else:
            if self.use_huber_loss:
                if self.use_value_active_masks:
                    critic_loss = huber_loss(error, self.huber_delta).sum() / ((1 - curr_env_dones) * (1 - curr_agent_dones)).sum()
                else:
                    critic_loss = huber_loss(error, self.huber_delta).sum() / (1 - curr_env_dones).sum()
            else:
                if self.use_value_active_masks:
                    critic_loss = mse_loss(error).sum() / ((1 - curr_env_dones) * (1 - curr_agent_dones)).sum()
                else:
                    critic_loss = mse_loss(error).sum() / (1 - curr_env_dones).sum()
            new_priorities = None

        update_policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.critic.parameters(),
                                                                 self.args.max_grad_norm)
        update_policy.critic_optimizer.step()

        # actor update: can form losses for each agent that the update policy controls
        # freeze Q-networks
        for p in update_policy.critic.parameters():
            p.requires_grad = False

        agent_Q_sequences = []
        # formulate mask to determine how to combine actor output actions with batch output actions
        mask_temp = []
        for p_id in self.policy_ids:
            if isinstance(self.policies[p_id].act_dim, np.ndarray):
                # multidiscrete case
                sum_act_dim = int(sum(self.policies[p_id].act_dim))
            else:
                sum_act_dim = self.policies[p_id].act_dim
            for a_id in self.policy_agents[p_id]:
                mask_temp.append(np.zeros(sum_act_dim, dtype=np.float32))

        masks = []
        done_mask = []
        sum_act_dim = None
        # need to iterate through agents, but only formulate masks at each step
        for i in range(num_update_agents):
            curr_mask_temp = copy.deepcopy(mask_temp)
            # set the mask to 1 at locations where the action should come from the actor output
            if isinstance(update_policy.act_dim, np.ndarray):
                # multidiscrete case
                sum_act_dim = int(sum(update_policy.act_dim))
            else:
                sum_act_dim = update_policy.act_dim
            curr_mask_temp[act_sequence_replace_ind_start + i] = np.ones(sum_act_dim, dtype=np.float32)
            curr_mask_vec = np.concatenate(curr_mask_temp)
            # expand this mask into the proper size
            curr_mask = np.tile(curr_mask_vec, (batch_size, 1))
            masks.append(curr_mask)

            # now collect agent dones
            if self.use_value_active_masks:
                agent_done_sequence = check(dones_batch[update_policy_id][i])
                done_mask.append(agent_done_sequence)
            else:
                agent_done_sequence = check(dones_batch[update_policy_id][i])
                agent_first_step_dones = torch.zeros((1, agent_done_sequence.shape[1], agent_done_sequence.shape[2]))
                agent_next_steps_dones = agent_done_sequence[: self.episode_length - 1, :, :]
                curr_agent_dones = torch.cat((agent_first_step_dones, agent_next_steps_dones), dim=0)
                done_mask.append(curr_agent_dones)

        # cat masks and form into torch tensors
        mask = check(np.concatenate(masks)).to(**self.tpdv)
        done_mask = torch.cat(done_mask, dim=1).to(**self.tpdv)

        # stack obs, acts, and available acts of all agents along batch dimension to process at once
        pol_prev_buffer_act_seq = np.concatenate((np.zeros((1, total_batch_size, sum_act_dim), dtype=np.float32),
                                                  np.concatenate(act_batch[update_policy_id][:, : -1], axis=1)))
        pol_agents_obs_seq = np.concatenate(obs_batch[update_policy_id], axis=1)
        if avail_act_batch[update_policy_id] is not None:
            pol_agents_avail_act_seq = np.concatenate(avail_act_batch[update_policy_id], axis=1)
        else:
            pol_agents_avail_act_seq = None
        # get all the actions from actor, with gumbel softmax to differentiate through the samples
        policy_act_seq, _, _ = update_policy.get_actions(pol_agents_obs_seq, pol_prev_buffer_act_seq,
                                                         update_policy.init_hidden(-1, total_batch_size),
                                                         available_actions=pol_agents_avail_act_seq,
                                                         use_gumbel=True)
        # separate the output into individual agent act sequences
        agent_actor_seqs = policy_act_seq.split(split_size=batch_size, dim=1)
        # convert act sequences to torch, formulate centralized buffer action, and repeat as done above
        act_sequences = list(map(lambda arr: check(arr).to(**self.tpdv), act_sequences))

        actor_cent_acts = copy.deepcopy(act_sequences)
        for i in range(num_update_agents):
            actor_cent_acts[act_sequence_replace_ind_start + i] = agent_actor_seqs[i]
        # cat these along final dim to formulate centralized action and stack copies of the batch so all agents can be updated
        actor_cent_acts = torch.cat(actor_cent_acts, dim=-1).repeat((1, num_update_agents, 1))

        batch_cent_acts = torch.cat(act_sequences, dim=-1).repeat((1, num_update_agents, 1))
        # also repeat the cent obs
        critic_rnn_state = update_policy.init_hidden(-1, total_batch_size)

        # iterate through timesteps and and get Q values to form actor loss
        for t in range(self.episode_length):
            # get Q values at timestep t with the replaced actions
            replaced_cent_act_batch = mask * actor_cent_acts[t] + (1 - mask) * batch_cent_acts[t]
            # get Q values at timestep but don't store the new RNN state
            Q_t, _ = update_policy.critic(all_agent_cent_obs[t], replaced_cent_act_batch, critic_rnn_state)
            # update the RNN state by stepping the RNN through with buffer sequence
            _, critic_rnn_state = update_policy.critic(all_agent_cent_obs[t], batch_cent_acts[t], critic_rnn_state)
            agent_Q_sequences.append(Q_t)
        # stack over time
        agent_Q_sequences = torch.stack(agent_Q_sequences)
        # mask at the places where agents were terminated in env
        agent_Q_sequences = agent_Q_sequences * (1 - done_mask)

        actor_loss = (-agent_Q_sequences).sum() / (1 - done_mask).sum()

        update_policy.critic_optimizer.zero_grad()
        update_policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.actor.parameters(),
                                                                self.args.max_grad_norm)
        update_policy.actor_optimizer.step()

        # unfreeze the Q networks
        for p in update_policy.critic.parameters():
            p.requires_grad = True

        train_info = {}
        
        train_info['critic_loss'] = critic_loss
        train_info['actor_loss'] = actor_loss
        train_info['critic_grad_norm'] = critic_grad_norm
        train_info['actor_grad_norm'] = actor_grad_norm

        return train_info, new_priorities, idxes

    def prep_training(self):
        for policy in self.policies.values():
            policy.actor.train()
            policy.critic.train()
            policy.target_actor.train()
            policy.target_critic.train()

    def prep_rollout(self):
        for policy in self.policies.values():
            policy.actor.eval()
            policy.critic.eval()
            policy.target_actor.eval()
            policy.target_critic.eval()
