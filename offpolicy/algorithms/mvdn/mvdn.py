import torch
import copy
from offpolicy.utils.util import make_onehot, soft_update, huber_loss, mse_loss
import numpy as np
from offpolicy.utils.popart import PopArt
from offpolicy.algorithms.mvdn.algorithm.mvdn_mixer import M_VDNMixer


class M_VDN:
    def __init__(self, args, num_agents, policies, policy_mapping_fn, device=torch.device("cuda:0")):
        """Class to do gradient updates"""
        self.args = args
        self.use_popart = self.args.use_popart
        self.use_value_active_masks = self.args.use_value_active_masks
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta

        self.device = device
        self.lr = self.args.lr
        self.tau = self.args.tau
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay

        self.num_agents = num_agents
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_ids = sorted(list(self.policies.keys()))
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in range(self.num_agents) if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}
        if self.use_popart:
            self.value_normalizer = {policy_id: PopArt(
                1) for policy_id in self.policies.keys()}

        multidiscrete_list = None
        if any([isinstance(policy.act_dim, np.ndarray) for policy in self.policies.values()]):
            # multidiscrete
            multidiscrete_list = [len(self.policies[p_id].act_dim) *
                                  len(self.policy_agents[p_id]) for p_id in self.policy_ids]

        # mixer network 
        self.mixer = M_VDNMixer(
            args, self.num_agents, self.policies['policy_0'].central_obs_dim, self.device, multidiscrete_list=multidiscrete_list)

        # target policies/networks
        self.target_policies = {p_id: copy.deepcopy(
            self.policies[p_id]) for p_id in self.policy_ids}
        self.target_mixer = copy.deepcopy(self.mixer)

        # collect all trainable parameters: each policy parameters, and the mixer parameters
        self.parameters = []
        for policy in self.policies.values():
            self.parameters += policy.parameters()
        self.parameters += self.mixer.parameters()

        self.optimizer = torch.optim.Adam(
            params=self.parameters, lr=self.lr, eps=self.opti_eps)

        if args.double_q:
            print("double Q learning will be used")

    def train_policy_on_batch(self, batch, use_same_share_obs):
        # unpack the batch
        obs_batch, cent_obs_batch, \
            act_batch, rew_batch, \
            nobs_batch, cent_nobs_batch, \
            dones_batch, dones_env_batch, \
            avail_act_batch, navail_act_batch, \
            importance_weights, idxes = batch

        if use_same_share_obs:
            cent_obs_batch = torch.FloatTensor(
                cent_obs_batch[self.policy_ids[0]])
            cent_nobs_batch = torch.FloatTensor(
                cent_nobs_batch[self.policy_ids[0]])
        else:
            choose_agent_id = 0
            cent_obs_batch = torch.FloatTensor(
                cent_obs_batch[self.policy_ids[0]][choose_agent_id])
            cent_nobs_batch = torch.FloatTensor(
                cent_nobs_batch[self.policy_ids[0]][choose_agent_id])

        dones_env_batch = torch.FloatTensor(
            dones_env_batch[self.policy_ids[0]])

        # individual agent q value sequences: each element is of shape (batch_size, 1)
        agent_q_sequences = []
        agent_next_q_sequences = []

        for p_id in self.policy_ids:
            policy = self.policies[p_id]
            target_policy = self.target_policies[p_id]
            # get data related to the policy id
            rewards = torch.FloatTensor(rew_batch[p_id][0])
            curr_obs_batch = torch.FloatTensor(obs_batch[p_id])
            curr_act_batch = torch.FloatTensor(act_batch[p_id])
            curr_nobs_batch = torch.FloatTensor(nobs_batch[p_id])

            # stacked_obs_batch size : [agent_num*batch_size, obs_shape]
            stacked_act_batch = torch.cat(list(curr_act_batch), dim=-2)
            stacked_obs_batch = torch.cat(list(curr_obs_batch), dim=-2)
            stacked_nobs_batch = torch.cat(list(curr_nobs_batch), dim=-2)

            if navail_act_batch[p_id] is not None:
                curr_navail_act_batch = torch.FloatTensor(
                    navail_act_batch[p_id])
                stacked_navail_act_batch = torch.cat(
                    list(curr_navail_act_batch), dim=-2)
            else:
                stacked_navail_act_batch = None

            # curr_obs_batch size : agent_num*batch_size*obs_shape
            batch_size = curr_obs_batch.shape[1]
            total_batch_size = batch_size * len(self.policy_agents[p_id])

            pol_all_q_out_sequence = policy.get_q_values(stacked_obs_batch)

            if isinstance(pol_all_q_out_sequence, list):
                # multidiscrete case
                ind = 0
                Q_per_part = []
                for i in range(len(policy.act_dim)):
                    curr_stacked_act_batch = stacked_act_batch[:,
                                                               ind: ind + policy.act_dim[i]]
                    curr_stacked_act_batch_ind = curr_stacked_act_batch.max(
                        dim=-1)[1]
                    curr_all_q_out_sequence = pol_all_q_out_sequence[i]
                    curr_pol_q_out_sequence = torch.gather(
                        curr_all_q_out_sequence, 1, curr_stacked_act_batch_ind.unsqueeze(dim=-1))
                    Q_per_part.append(curr_pol_q_out_sequence)
                    ind += policy.act_dim[i]
                Q_sequence_combined_parts = torch.cat(Q_per_part, dim=-1)
                pol_agents_q_out_sequence = Q_sequence_combined_parts.split(
                    split_size=batch_size, dim=-2)
            else:
                # get the q values associated with the action taken acording ot the batch
                stacked_act_batch_ind = stacked_act_batch.max(dim=-1)[1]
                # pol_q_out_sequence : batch_size * 1
                pol_q_out_sequence = torch.gather(
                    pol_all_q_out_sequence, 1, stacked_act_batch_ind.unsqueeze(dim=-1))
                # separate into agent q sequences for each agent, then cat along the final dimension to prepare for mixer input
                pol_agents_q_out_sequence = pol_q_out_sequence.split(
                    split_size=batch_size, dim=-2)

            agent_q_sequences.append(
                torch.cat(pol_agents_q_out_sequence, dim=-1))

            with torch.no_grad():
                if self.args.double_q:
                    # actions come from live q; get the q values for the final nobs
                    pol_final_qs = policy.get_q_values(stacked_nobs_batch[-1])

                    if type(pol_final_qs) == list:
                        # multidiscrete case
                        assert stacked_navail_act_batch is None, "Available actions not supported for multidiscrete"
                        pol_nacts = []
                        for i in range(len(pol_final_qs)):
                            pol_final_curr_qs = pol_final_qs[i]
                            pol_all_curr_q_out_seq = pol_all_q_out_sequence[i]
                            pol_all_nq_out_curr_seq = torch.cat(
                                (pol_all_curr_q_out_seq[1:], pol_final_curr_qs[None]))
                            pol_curr_nacts = pol_all_nq_out_curr_seq.max(
                                dim=-1)[1]
                            pol_nacts.append(pol_curr_nacts)
                        #pol_nacts = np.concatenate(pol_nacts, axis=-1)
                        targ_pol_nq_seq = target_policy.get_q_values(
                            stacked_nobs_batch, action_batch=pol_nacts)
                    else:
                        # cat to form all the next step qs
                        pol_all_nq_out_sequence = torch.cat(
                            (pol_all_q_out_sequence[1:], pol_final_qs[None]))
                        # mask out the unavailable actions
                        if stacked_navail_act_batch is not None:
                            pol_all_nq_out_sequence[stacked_navail_act_batch == 0.0] = -1e10
                        # greedily choose actions which maximize the q values and convert these actions to onehot
                        pol_nacts = pol_all_nq_out_sequence.max(dim=-1)[1]
                        # q values given by target but evaluated at actions taken by live
                        targ_pol_nq_seq = target_policy.get_q_values(
                            stacked_nobs_batch, action_batch=pol_nacts)
                else:
                    # just choose actions from target policy
                    _, targ_pol_nq_seq = target_policy.get_actions(stacked_nobs_batch,
                                                                   available_actions=stacked_navail_act_batch,
                                                                   t_env=None,
                                                                   explore=False)
                    targ_pol_nq_seq = targ_pol_all_nq_seq.max(dim=-1)[0]
                    targ_pol_nq_seq = targ_pol_nq_seq.unsqueeze(-1)
                # separate the next qs into sequences for each agent
                pol_agents_nq_sequence = targ_pol_nq_seq.split(
                    split_size=batch_size, dim=-2)
            # cat target qs along the final dim
            agent_next_q_sequences.append(
                torch.cat(pol_agents_nq_sequence, dim=-1))
        # combine the agent q value sequences to feed into mixer networks
        agent_q_sequences = torch.cat(agent_q_sequences, dim=-1)
        agent_next_q_sequences = torch.cat(agent_next_q_sequences, dim=-1)

        # store the sequences of predicted and next step Q_tot values to form Bellman errors
        predicted_Q_tot_vals = []
        next_step_Q_tot_vals = []

        curr_Q_tot = self.mixer(agent_q_sequences, cent_obs_batch)
        next_step_Q_tot = self.target_mixer(
            agent_next_q_sequences, cent_nobs_batch)

        predicted_Q_tot_vals.append(curr_Q_tot.squeeze(-1))
        next_step_Q_tot_vals.append(next_step_Q_tot.squeeze(-1))

        # stack over time dimension
        predicted_Q_tot_vals = torch.stack(predicted_Q_tot_vals)
        next_step_Q_tot_vals = torch.stack(next_step_Q_tot_vals)

        # all agents must share reward, so get the reward sequence for an agent
        # form bootstrapped targets
        if self.use_popart:
            Q_tot_targets = rewards + (1 - dones_env_batch) * self.args.gamma * \
                self.value_normalizer[p_id].denormalize(next_step_Q_tot_vals)
            Q_tot_targets = self.value_normalizer[p_id](Q_tot_targets)
        else:
            Q_tot_targets = rewards + \
                (1 - dones_env_batch) * self.args.gamma * next_step_Q_tot_vals
        # form mask to mask out sequence elements corresponding to states at which the episode already ended
        predicted_Q_tots = predicted_Q_tot_vals

        if self.use_value_active_masks:
            curr_agent_dones = torch.FloatTensor(
                dones_batch[p_id][choose_agent_id])
            predicted_Q_tots = predicted_Q_tots * (1 - curr_agent_dones)
            Q_tot_targets = Q_tot_targets * (1 - curr_agent_dones)

        # loss is MSE Bellman Error
        error = predicted_Q_tots - Q_tot_targets.detach()
        if self.use_per:
            if self.use_huber_loss:
                loss = huber_loss(error, self.huber_delta).flatten()
            else:
                loss = mse_loss(error).flatten()
            loss = (loss * torch.FloatTensor(importance_weights)).mean()
            # new priorities are a combination of the maximum TD error across sequence and the mean TD error across sequence
            new_priorities = error.abs().detach().numpy().flatten() + self.per_eps
        else:
            if self.use_huber_loss:
                loss = huber_loss(error, self.huber_delta).mean()
            else:
                if self.use_value_active_masks:
                    loss = mse_loss(error).sum() / (1 - curr_agent_dones).sum()
                else:
                    loss = mse_loss(error).mean()
            new_priorities = None

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters, self.args.max_grad_norm)
        self.optimizer.step()

        return (loss, grad_norm, predicted_Q_tots.mean()), new_priorities, idxes

    def hard_target_updates(self):
        print("hard update targets")
        for policy_id in self.policy_ids:
            self.target_policies[policy_id].load_state(
                self.policies[policy_id])
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def soft_target_updates(self):
        print("soft update targets")
        for policy_id in self.policy_ids:
            soft_update(
                self.target_policies[policy_id], self.policies[policy_id], self.tau)
        if self.mixer is not None:
            soft_update(self.target_mixer, self.mixer, self.tau)

    def prep_training(self):
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.train()
            self.target_policies[p_id].q_network.train()
        self.mixer.train()
        self.target_mixer.train()

    def prep_rollout(self):
        for p_id in self.policy_ids:
            self.policies[p_id].q_network.eval()
            self.target_policies[p_id].q_network.eval()
        self.mixer.eval()
        self.target_mixer.eval()
