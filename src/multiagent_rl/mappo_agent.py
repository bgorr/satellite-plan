import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.local_states = []
        self.full_states = []
        self.probs = [] # log probs
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = [] # terminal flags

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.local_states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.local_states), np.array(self.full_states), np.array(self.actions),\
                np.array(self.probs), np.array(self.vals),\
                np.array(self.rewards), np.array(self.dones),\
                batches
    
    def store_memory(self, local_state, full_state, action, probs, vals, reward, done):
        self.local_states.append(local_state)
        self.full_states.append(full_state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.local_states = []
        self.full_states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, settings, n_sats, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        self.settings = settings
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_mappo')
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.n_sats = n_sats
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        norm_tensor_array = []
        norm_tensor_array.append(86400*self.settings["time"]["duration"]/self.settings["time"]["step_size"])
        norm_tensor_array.append(self.settings["instrument"]["ffor"]/2)
        norm_tensor_array.append(180)
        norm_tensor_array.append(180)
        norm_tensor_array.append(self.n_sats)

        state = state / T.tensor(norm_tensor_array, dtype=T.float).to(self.device)
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, settings, n_sats, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()
        self.settings = settings
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_mappo')
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.n_sats = n_sats

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        norm_tensor_array = []
        for i in range(self.n_sats):
            norm_tensor_array.append(86400*self.settings["time"]["duration"]/self.settings["time"]["step_size"])
            norm_tensor_array.append(self.settings["instrument"]["ffor"]/2)
            norm_tensor_array.append(180)
            norm_tensor_array.append(180)
            norm_tensor_array.append(self.n_sats)

        state = state / T.tensor(norm_tensor_array, dtype=T.float).to(self.device)
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, settings, num_sats, n_actions, actor_input_dims, critic_input_dims, gamma=0.99, alpha=3e-4, gae_lambda=0.97, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(settings, num_sats, n_actions, actor_input_dims, alpha)
        self.critic = CriticNetwork(settings, num_sats, critic_input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, local_state, full_state, action, probs, vals, reward, done):
        self.memory.store_memory(local_state, full_state, action, probs, vals, reward, done)

    def save_models(self):
        print('saving models')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('loading models')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, local_obs, full_obs):
        local_state = T.tensor([local_obs], dtype=T.float).to(self.actor.device)
        full_state = T.tensor([full_obs], dtype=T.float).to(self.actor.device)

        dist = self.actor(local_state)
        value = self.critic(full_state)
        action = dist.sample()

        prob = T.squeeze(dist.log_prob(action)).item()
        # probs = []
        # for i in range(5):
        #     probs.append(T.squeeze(dist.log_prob(T.tensor(i))).item())
        # print(probs)
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            local_state_arr, full_state_arr, action_arr, old_probs_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
        
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                        (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            kl_divs = []
            for batch in batches:
                local_states = T.tensor(local_state_arr[batch], dtype=T.float).to(self.actor.device)
                full_states = T.tensor(full_state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(local_states)
                critic_value = self.critic(full_states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                
                #total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                #total_loss.backward()
                actor_loss.backward()
                critic_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            #     kl_divs.append((old_probs - new_probs).mean().detach().numpy())
            # target_kl = 0.01
            # kl = np.sum(kl_divs)
            # if kl > 1.5 * target_kl:
            #     break

        self.memory.clear_memory()

