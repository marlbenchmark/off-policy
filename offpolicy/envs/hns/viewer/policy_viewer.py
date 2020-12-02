#!/usr/bin/env python
import time
import glfw
import numpy as np
from operator import itemgetter
from mujoco_py import const, MjViewer
from mujoco_worldgen.util.types import store_args
from envs.hns.ma_policy.util import listdict2dictnp
import pdb
import torch


def splitobs(obs, keepdims=True):
    '''
        Split obs into list of single agent obs.
        Args:
            obs: dictionary of numpy arrays where first dim in each array is agent dim
    '''
    n_agents = obs[list(obs.keys())[0]].shape[0]
    return [{k: v[[i]] if keepdims else v[i] for k, v in obs.items()} for i in range(n_agents)]


class PolicyViewer(MjViewer):
    '''
    PolicyViewer runs a policy with an environment and optionally displays it.
        env - environment to run policy in
        policy - policy object to run
        display_window - if true, show the graphical viewer
        seed - environment seed to view
        duration - time in seconds to run the policy, run forever if duration=None
    '''
    @store_args
    def __init__(self, env, policies, display_window=True, seed=None, duration=None):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)
        self.total_rew = 0.0
        self.ob = env.reset()
        for policy in self.policies:
            policy.reset()
        assert env.metadata['n_actors'] % len(policies) == 0
        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)
        # TO DO: remove circular dependency on viewer object. It looks fishy.
        self.env.unwrapped.viewer = self
        if self.render and self.display_window:
            self.env.render()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        # Increment experiment seed
        if key == glfw.KEY_N:
            self.reset_increment()
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            print("Pressed P")
            self.seed = max(self.seed - 1, 0)
            self.env.seed(self.seed)
            self.ob = self.env.reset()
            for policy in self.policies:
                policy.reset()
            if hasattr(self.env, "reset_goal"):
                self.goal = self.env.reset_goal()
            self.update_sim(self.env.unwrapped.sim)

    def run(self):
        if self.duration is not None:
            self.end_time = time.time() + self.duration
        self.total_rew_avg = 0.0
        self.n_episodes = 0
        while self.duration is None or time.time() < self.end_time:
            if len(self.policies) == 1:
                action, _ = self.policies[0].act(self.ob)
            else:
                self.ob = splitobs(self.ob, keepdims=False)
                ob_policy_idx = np.split(np.arange(len(self.ob)), len(self.policies))
                actions = []
                for i, policy in enumerate(self.policies):
                    inp = itemgetter(*ob_policy_idx[i])(self.ob)
                    inp = listdict2dictnp([inp] if ob_policy_idx[i].shape[0] == 1 else inp)
                    ac, info = policy.act(inp)
                    actions.append(ac)
                action = listdict2dictnp(actions, keepdims=True)
            
            self.ob, rew, done, env_info = self.env.step(action)
            self.total_rew += rew

            if done or env_info.get('discard_episode', False):
                self.reset_increment()

            if self.display_window:
                self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                self.add_overlay(const.GRID_TOPRIGHT, "Reward", str(self.total_rew))
                if hasattr(self.env.unwrapped, "viewer_stats"):
                    for k, v in self.env.unwrapped.viewer_stats.items():
                        self.add_overlay(const.GRID_TOPRIGHT, k, str(v))

                self.env.render()

    def reset_increment(self):
        self.total_rew_avg = (self.n_episodes * self.total_rew_avg + self.total_rew) / (self.n_episodes + 1)
        self.n_episodes += 1
        print(f"Reward: {self.total_rew} (rolling average: {self.total_rew_avg})")
        self.total_rew = 0.0
        self.seed += 1
        self.env.seed(self.seed)
        self.ob = self.env.reset()
        for policy in self.policies:
            policy.reset()
        if hasattr(self.env, "reset_goal"):
            self.goal = self.env.reset_goal()
        self.update_sim(self.env.unwrapped.sim)

class PolicyViewer_hs(MjViewer):
    '''
    PolicyViewer runs a policy with an environment and optionally displays it.
        env - environment to run policy in
        policy - policy object to run
        display_window - if true, show the graphical viewer
        seed - environment seed to view
        duration - time in seconds to run the policy, run forever if duration=None
    '''
    @store_args
    def __init__(self, env, policies, display_window=True, seed=None, duration=None):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)
        self.total_rew = 0.0
        self.dict_obs = env.reset()
        #for policy in self.policies:
        #    policy.reset()
        assert env.metadata['n_actors'] % len(policies) == 0
        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)
        # TO DO: remove circular dependency on viewer object. It looks fishy.
        self.env.unwrapped.viewer = self
        if self.render and self.display_window:
            self.env.render()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        # Increment experiment seed
        if key == glfw.KEY_N:
            self.reset_increment()
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            print("Pressed P")
            self.seed = max(self.seed - 1, 0)
            self.env.seed(self.seed)
            self.ob = self.env.reset()
            for policy in self.policies:
                policy.reset()
            if hasattr(self.env, "reset_goal"):
                self.goal = self.env.reset_goal()
            self.update_sim(self.env.unwrapped.sim)

    def run(self):
        self.action_movement_dim = []
        '''
        self.order_obs = ['agent_qpos_qvel','box_obs','ramp_obs','food_obs','observation_self']    
        self.mask_order_obs = ['mask_aa_obs','mask_ab_obs','mask_ar_obs','mask_af_obs',None]
        '''
        self.order_obs = ['box_obs','ramp_obs','construction_site_obs','observation_self']   
        self.mask_order_obs =  ['mask_ab_obs','mask_ar_obs',None,None]
   
        self.num_agents = 2
        for agent_id in range(self.num_agents):
            # deal with dict action space
            action_movement = self.env.action_space['action_movement'][agent_id].nvec
            self.action_movement_dim.append(len(action_movement))
        self.masks = np.ones((1, self.num_agents, 1)).astype(np.float32)
        if self.duration is not None:
            self.end_time = time.time() + self.duration
        self.total_rew_avg = 0.0
        self.n_episodes = 0
        self.obs = []
        self.share_obs = []   
        for i, key in enumerate(self.order_obs):
            if key in self.env.observation_space.spaces.keys():             
                if self.mask_order_obs[i] == None:          
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = self.dict_obs[key].copy()
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(self.num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32) 
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        while self.duration is None or time.time() < self.end_time:
            values = []
            actions= []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
            with torch.no_grad():                
                for agent_id in range(self.num_agents):
                    self.policies[0].eval()
                    print(self.recurrent_hidden_states)
                    print(type(self.recurrent_hidden_states))
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = self.policies[0].act(agent_id,
                    torch.tensor(self.share_obs[:,agent_id,:]), 
                    torch.tensor(self.obs[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states_critic[:,agent_id,:]),
                    torch.tensor(self.masks[:,agent_id,:]))
                    values.append(value.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())

            action_movement = []
            action_pull = []
            action_glueall = []
            for agent_id in range(self.num_agents):
                action_movement.append(actions[agent_id][0][:self.action_movement_dim[agent_id]])
                action_glueall.append(int(actions[agent_id][0][self.action_movement_dim[agent_id]]))
                if 'action_pull' in self.env.action_space.spaces.keys():
                    action_pull.append(int(actions[agent_id][-1]))
            action_movement = np.stack(action_movement, axis = 0)
            action_glueall = np.stack(action_glueall, axis = 0)
            if 'action_pull' in self.env.action_space.spaces.keys():
                action_pull = np.stack(action_pull, axis = 0)                             
            one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}
        
            self.dict_obs, rew, done, env_info = self.env.step(one_env_action)
            self.total_rew += rew
            self.obs = []
            self.share_obs = []   
            for i, key in enumerate(self.order_obs):
                if key in self.env.observation_space.spaces.keys():             
                    if self.mask_order_obs[i] == None:
                        temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                        temp_obs = temp_share_obs.copy()
                    else:
                        temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                        temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                        temp_obs = self.dict_obs[key].copy()
                        mins_temp_mask = ~temp_mask
                        temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                        temp_obs = temp_obs.reshape(self.num_agents,-1) 
                    if i == 0:
                        reshape_obs = temp_obs.copy()
                        reshape_share_obs = temp_share_obs.copy()
                    else:
                        reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                        reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
            self.obs.append(reshape_obs)
            self.share_obs.append(reshape_share_obs)   
            self.obs = np.array(self.obs).astype(np.float32)
            self.share_obs = np.array(self.share_obs).astype(np.float32)
            self.recurrent_hidden_states = np.array(recurrent_hidden_statess).transpose(1,0,2)
            self.recurrent_hidden_states_critic = np.array(recurrent_hidden_statess_critic).transpose(1,0,2)
            if done or env_info.get('discard_episode', False):
                self.reset_increment()

            if self.display_window:
                self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                self.add_overlay(const.GRID_TOPRIGHT, "Reward", str(self.total_rew))
                if hasattr(self.env.unwrapped, "viewer_stats"):
                    for k, v in self.env.unwrapped.viewer_stats.items():
                        self.add_overlay(const.GRID_TOPRIGHT, k, str(v))

                self.env.render()

    def reset_increment(self):
        self.total_rew_avg = (self.n_episodes * self.total_rew_avg + self.total_rew) / (self.n_episodes + 1)
        self.n_episodes += 1
        print(f"Reward: {self.total_rew} (rolling average: {self.total_rew_avg})")
        self.total_rew = 0.0
        self.seed += 1
        self.env.seed(self.seed)
        self.dict_obs = self.env.reset()
        self.obs = []
        self.share_obs = []   
        for i, key in enumerate(self.order_obs):
            if key in self.env.observation_space.spaces.keys():             
                if self.mask_order_obs[i] == None:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = self.dict_obs[key].copy()
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(self.num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32) 
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        #for policy in self.policies:
        #    policy.reset()
        if hasattr(self.env, "reset_goal"):
            self.goal = self.env.reset_goal()
        self.update_sim(self.env.unwrapped.sim)