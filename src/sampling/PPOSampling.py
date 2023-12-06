import numpy as np
from copy import deepcopy
import scipy.signal
import tensorflow as tf
import config

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



class PPOSampling:

    def __init__(self, satellites):
        self.episode_steps = config.ppo_episode_steps
        self.batch_size = config.ppo_batch_size
        self.satellites = satellites
        self.num_reels = min([len(sat['experience_reels']) for sat in self.satellites])
        self.longest_reel = max([max([len(reel) for reel in sat['experience_reels']]) for sat in self.satellites])
        self.gamma = 0.99
        self.lam = 0.97

    def sample_old(self):
        sat_trajectories = []
        sat_trajectory_critic_values = []
        # if self.num_reels < self.batch_size:
        #     return sat_trajectories, sat_trajectory_critic_values, [], []

        rand_reels = []
        for x in range(self.batch_size):
            rand_reels.append(self.num_reels - (x + 1))
        # rand_reels = np.random.randint(0, self.num_reels, size=self.batch_size).tolist()
        # rand_reels = [self.num_reels - 1]

        num_reels = len(self.satellites[0]['experience_reels'])
        num_sats = len(self.satellites)

        sat_reel_mins = [1000000] * num_reels
        for sat in self.satellites:
            for idx, experience_reel in enumerate(sat['experience_reels']):
                reel_len = len(experience_reel)
                if reel_len < sat_reel_mins[idx]:
                    sat_reel_mins[idx] = reel_len



        total_advantages = []
        total_returns = []
        total_log_probs = []
        total_actions = []
        total_states = []  # (batch_size, )

        reward_reels_t = []
        log_probs_s = []
        actions_s = []
        states_s = []
        advantages_s = []
        for _ in range(num_reels):
            reward_reels_t.append([])
        for _ in range(num_sats):
            log_probs_s.append([])
            actions_s.append([])
            states_s.append([])
            advantages_s.append([])



        for sat_idx, sat in enumerate(self.satellites):  # over: satellites
            # sat_samples = [sat['experience_reels'][r] for r in rand_reels]
            # sat_critic_samples = [sat['critic_reels'][r] for r in rand_reels]
            sat_samples = [r for r in sat['experience_reels']]
            sat_critic_samples = [r for r in sat['critic_reels']]

            sat_advantages = []
            sat_returns = []  # (batch_size, trajectory_len)
            sat_log_probs = []
            sat_actions = []
            sat_states = []

            reel_idx = -1
            for critic_reel, sat_sample in zip(sat_critic_samples, sat_samples):  # over: satellite trajectories
                reel_idx += 1

                # Reels
                state_reel = [samp[0] for samp in sat_sample]  # (trajectory_length, state_dim)
                action_reel = [samp[1] for samp in sat_sample]  # (trajectory_length, 1)
                reward_reel = [samp[2] for samp in sat_sample]  # (trajectory_length, 1)
                log_prob_reel = [samp[4] for samp in sat_sample]  # (trajectory_length, 1)
                critic_reel_cp = deepcopy(critic_reel)  # shape: (235, 1, 1)

                # Padding
                # for x in range(self.episode_steps - len(state_reel)):
                #     state_reel.append([0.0 for _ in range(len(state_reel[0]))])
                # action_reel += [0] * (self.episode_steps - len(action_reel))
                # reward_reel += [0.0] * (self.episode_steps - len(reward_reel))
                # log_prob_reel += [0.0] * (self.episode_steps - len(log_prob_reel))
                # critic_reel_cp += [0.0] * (self.episode_steps - len(critic_reel_cp))

                # Clipping
                state_reel = state_reel[:sat_reel_mins[reel_idx]]
                action_reel = action_reel[:sat_reel_mins[reel_idx]]
                reward_reel = reward_reel[:sat_reel_mins[reel_idx]]
                log_prob_reel = log_prob_reel[:sat_reel_mins[reel_idx]]
                critic_reel_cp = critic_reel_cp[:sat_reel_mins[reel_idx]]

                reward_reels_t[reel_idx].append(reward_reel)
                log_probs_s[sat_idx].append(log_prob_reel)
                actions_s[sat_idx].append(action_reel)
                states_s[sat_idx].append(state_reel)

                state_tensor = tf.convert_to_tensor(state_reel, dtype=tf.float32)
                log_prob_tensor = tf.convert_to_tensor(log_prob_reel, dtype=tf.float32)
                action_tensor = tf.convert_to_tensor(action_reel, dtype=tf.int32)

                sat_log_probs.append(log_prob_tensor)
                sat_states.append(state_tensor)
                sat_actions.append(action_tensor)

            total_states.append(sat_states)
            total_log_probs.append(sat_log_probs)
            total_actions.append(sat_actions)


        # Critic inputs
        critic_inputs = []  # (batch_size, trajectory_len, num_agents * state_dim) --> (1, 499, 3)
        flat_critic_inputs = []
        for reel_idx in range(num_reels):
            batch_item_inputs = []  # (trajectory_len, num_agents * state_dim)
            for step_idx in range(sat_reel_mins[reel_idx]):
                combined_states = []  # (num_agents * state_dim)
                for sat_idx, sat in enumerate(self.satellites):
                    sat_state = total_states[sat_idx][reel_idx][step_idx]  # (trajectory_len, state_dim)
                    combined_states.extend(sat_state)
                batch_item_inputs.append(combined_states)
                flat_critic_inputs.append(combined_states)
            critic_inputs.append(tf.convert_to_tensor(batch_item_inputs, dtype=tf.float32))
        flat_critic_inputs = tf.convert_to_tensor(flat_critic_inputs, dtype=tf.float32)

        # Critic labels
        summed_traj_rewards = []
        for idx, reel in enumerate(reward_reels_t):
            summed_traj_reward = [sum(elements) for elements in zip(*reel)]
            summed_traj_reward.append(0)
            summed_traj_rewards.append(summed_traj_reward)
        summed_traj_rewards_copy = deepcopy(summed_traj_rewards)

        return_reels = []
        flat_critic_labels = []
        for reel in summed_traj_rewards:
            reel_return = discounted_cumulative_sums(reel, config.ppo_gamma)[:-1]
            return_reels.append(reel_return)
            flat_critic_labels.extend(reel_return)
        flat_critic_labels = tf.convert_to_tensor(flat_critic_labels, dtype=tf.float32)

        # Critic reels
        critic_reels = self.satellites[0]['critic_reels']
        clipped_critic_reels = []
        for reel_idx, c_reel in enumerate(critic_reels):
            clipped_reel = c_reel[:sat_reel_mins[reel_idx]]
            clipped_reel.append(0)
            clipped_critic_reels.append(clipped_reel)

        # Find satellite advantages (same for each satellite)
        reel_advantages = []
        reel_advantages_flat = []
        for reel_idx in range(num_reels):
            reward_reel = np.array(summed_traj_rewards_copy[reel_idx])
            value_reel = np.array(clipped_critic_reels[reel_idx])
            reel_deltas = reward_reel[:-1] + config.ppo_gamma * value_reel[1:] - value_reel[:-1]
            reel_advantage = discounted_cumulative_sums(reel_deltas, config.ppo_gamma * config.ppo_lambda)
            reel_advantages.extend(reel_advantage)
        reel_advantages = np.array(reel_advantages)
        advantage_mean, advantage_std = (
            np.mean(reel_advantages),
            np.std(reel_advantages),
        )
        reel_advantages = (reel_advantages - advantage_mean) / advantage_std
        reel_advantages = tf.convert_to_tensor(reel_advantages, dtype=tf.float32)


        return total_states, total_actions, total_log_probs, reel_advantages, flat_critic_inputs, flat_critic_labels





        #         # Compute advantages
        #         deltas = np.array(reward_reel[:-1]) + self.gamma * np.array(critic_reel_cp[1:]) - np.array(critic_reel_cp[:-1])
        #         advantages = discounted_cumulative_sums(
        #             deltas, self.gamma * self.lam
        #         )
        #         adv_mean, adv_std = (
        #             np.mean(advantages),
        #             np.std(advantages)
        #         )
        #         advantages = (advantages - adv_mean) / adv_std
        #
        #         # Compute returns
        #         returns = discounted_cumulative_sums(
        #             reward_reel, self.gamma
        #         )[:-1]
        #
        #         # Tensorize
        #         state_tensor = tf.convert_to_tensor(state_reel[:-1], dtype=tf.float32)
        #         log_prob_tensor = tf.convert_to_tensor(log_prob_reel[:-1], dtype=tf.float32)
        #         action_tensor = tf.convert_to_tensor(action_reel[:-1], dtype=tf.int32)
        #
        #         advantage_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        #         returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        #         # print(state_tensor.shape)      # shape: (500, 3)
        #         # print(advantage_tensor.shape)  # shape: (499)
        #         # print(returns.shape)           # shape: (499)
        #         # exit(0)
        #
        #         # Record
        #         sat_states.append(state_tensor)
        #         sat_log_probs.append(log_prob_tensor)
        #         sat_advantages.append(advantage_tensor)
        #         sat_returns.append(returns_tensor)
        #         sat_actions.append(action_tensor)
        #
        #
        #     # Tensorize
        #     sat_returns_tensor = tf.convert_to_tensor(sat_returns, dtype=tf.float32)
        #
        #
        #     total_advantages.append(sat_advantages)
        #     total_returns.append(sat_returns_tensor)
        #     total_log_probs.append(sat_log_probs)
        #     total_actions.append(sat_actions)
        #     total_states.append(sat_states)
        #
        #     sat_trajectories.append(deepcopy([sat['experience_reels'][r] for r in rand_reels]))
        #     sat_trajectory_critic_values.append(deepcopy([sat['critic_reels'][r] for r in rand_reels]))
        #
        # # shared critic inputs
        # critic_returns = tf.zeros_like(total_returns[0])  # shape: (batch_size, trajectory_len) --> (1, 499)
        # for sat_returns in total_returns:
        #     critic_returns += sat_returns

        # return sat_trajectories, sat_trajectory_critic_values, total_advantages, critic_returns, total_log_probs, total_actions, total_states



    def sample(self):
        sat_trajectories = []
        sat_trajectory_critic_values = []
        # if self.num_reels < self.batch_size:
        #     return sat_trajectories, sat_trajectory_critic_values, [], []

        rand_reels = []
        for x in range(self.batch_size):
            rand_reels.append(self.num_reels - (x + 1))
        # rand_reels = np.random.randint(0, self.num_reels, size=self.batch_size).tolist()
        # rand_reels = [self.num_reels - 1]

        num_reels = len(self.satellites[0]['experience_reels'])
        num_sats = len(self.satellites)

        sat_reel_mins = [1000000] * num_reels
        for sat in self.satellites:
            for idx, experience_reel in enumerate(sat['experience_reels']):
                reel_len = len(experience_reel)
                if reel_len < sat_reel_mins[idx]:
                    sat_reel_mins[idx] = reel_len



        total_advantages = []
        total_returns = []
        total_log_probs = []
        total_actions = []
        total_states = []  # (batch_size, )

        reward_reels_t = []
        log_probs_s = []
        actions_s = []
        states_s = []
        advantages_s = []
        for _ in range(num_reels):
            reward_reels_t.append([])
        for _ in range(num_sats):
            log_probs_s.append([])
            actions_s.append([])
            states_s.append([])
            advantages_s.append([])



        for sat_idx, sat in enumerate(self.satellites):  # over: satellites
            # sat_samples = [sat['experience_reels'][r] for r in rand_reels]
            # sat_critic_samples = [sat['critic_reels'][r] for r in rand_reels]
            sat_samples = [r for r in sat['experience_reels']]
            sat_critic_samples = [r for r in sat['critic_reels']]

            sat_advantages = []
            sat_returns = []  # (batch_size, trajectory_len)
            sat_log_probs = []
            sat_actions = []
            sat_states = []

            reel_idx = -1
            for critic_reel, sat_sample in zip(sat_critic_samples, sat_samples):  # over: satellite trajectories
                reel_idx += 1

                # Reels
                state_reel = [samp[0] for samp in sat_sample]  # (trajectory_length, state_dim)
                action_reel = [samp[1] for samp in sat_sample]  # (trajectory_length, 1)
                reward_reel = [samp[2] for samp in sat_sample]  # (trajectory_length, 1)
                log_prob_reel = [samp[4] for samp in sat_sample]  # (trajectory_length, 1)
                critic_reel_cp = deepcopy(critic_reel)  # shape: (235, 1, 1)

                # Padding
                # for x in range(self.episode_steps - len(state_reel)):
                #     state_reel.append([0.0 for _ in range(len(state_reel[0]))])
                # action_reel += [0] * (self.episode_steps - len(action_reel))
                # reward_reel += [0.0] * (self.episode_steps - len(reward_reel))
                # log_prob_reel += [0.0] * (self.episode_steps - len(log_prob_reel))
                # critic_reel_cp += [0.0] * (self.episode_steps - len(critic_reel_cp))

                # Clipping
                state_reel = state_reel[:sat_reel_mins[reel_idx]]
                action_reel = action_reel[:sat_reel_mins[reel_idx]]
                reward_reel = reward_reel[:sat_reel_mins[reel_idx]]
                log_prob_reel = log_prob_reel[:sat_reel_mins[reel_idx]]
                critic_reel_cp = critic_reel_cp[:sat_reel_mins[reel_idx]]

                reward_reels_t[reel_idx].append(reward_reel)
                log_probs_s[sat_idx].append(log_prob_reel)
                actions_s[sat_idx].append(action_reel)
                states_s[sat_idx].append(state_reel)

                state_tensor = tf.convert_to_tensor(state_reel, dtype=tf.float32)
                log_prob_tensor = tf.convert_to_tensor(log_prob_reel, dtype=tf.float32)
                action_tensor = tf.convert_to_tensor(action_reel, dtype=tf.int32)

                sat_log_probs.append(log_prob_tensor)
                sat_states.append(state_tensor)
                sat_actions.append(action_tensor)

            total_states.append(sat_states)
            total_log_probs.append(sat_log_probs)
            total_actions.append(sat_actions)


        # Critic inputs
        critic_inputs = []  # (batch_size, trajectory_len, num_agents * state_dim) --> (1, 499, 3)
        flat_critic_inputs = []
        for reel_idx in range(num_reels):
            batch_item_inputs = []  # (trajectory_len, num_agents * state_dim)
            for step_idx in range(sat_reel_mins[reel_idx]):
                combined_states = []  # (num_agents * state_dim)
                for sat_idx, sat in enumerate(self.satellites):
                    sat_state = total_states[sat_idx][reel_idx][step_idx]  # (trajectory_len, state_dim)
                    combined_states.extend(sat_state)
                batch_item_inputs.append(combined_states)
                flat_critic_inputs.append(combined_states)
            critic_inputs.append(tf.convert_to_tensor(batch_item_inputs, dtype=tf.float32))
        flat_critic_inputs = tf.convert_to_tensor(flat_critic_inputs, dtype=tf.float32)

        # Critic labels
        summed_traj_rewards = []
        for idx, reel in enumerate(reward_reels_t):
            summed_traj_reward = [sum(elements) for elements in zip(*reel)]
            summed_traj_reward.append(0)
            summed_traj_rewards.append(summed_traj_reward)
        summed_traj_rewards_copy = deepcopy(summed_traj_rewards)

        return_reels = []
        flat_critic_labels = []
        for reel in summed_traj_rewards:
            reel_return = discounted_cumulative_sums(reel, config.ppo_gamma)[:-1]
            return_reels.append(reel_return)
            flat_critic_labels.extend(reel_return)
        flat_critic_labels = tf.convert_to_tensor(flat_critic_labels, dtype=tf.float32)

        # Critic reels
        critic_reels = self.satellites[0]['critic_reels']
        clipped_critic_reels = []
        for reel_idx, c_reel in enumerate(critic_reels):
            clipped_reel = c_reel[:sat_reel_mins[reel_idx]]
            clipped_reel.append(0)
            clipped_critic_reels.append(clipped_reel)

        # Find satellite advantages (same for each satellite)
        reel_advantages = []
        reel_advantages_flat = []
        for reel_idx in range(num_reels):
            reward_reel = np.array(summed_traj_rewards_copy[reel_idx])
            value_reel = np.array(clipped_critic_reels[reel_idx])
            reel_deltas = reward_reel[:-1] + config.ppo_gamma * value_reel[1:] - value_reel[:-1]
            reel_advantage = discounted_cumulative_sums(reel_deltas, config.ppo_gamma * config.ppo_lambda)
            reel_advantages.extend(reel_advantage)
        reel_advantages = np.array(reel_advantages)
        advantage_mean, advantage_std = (
            np.mean(reel_advantages),
            np.std(reel_advantages),
        )
        reel_advantages = (reel_advantages - advantage_mean) / advantage_std
        reel_advantages = tf.convert_to_tensor(reel_advantages, dtype=tf.float32)


        return total_states, total_actions, total_log_probs, reel_advantages, flat_critic_inputs, flat_critic_labels





        #         # Compute advantages
        #         deltas = np.array(reward_reel[:-1]) + self.gamma * np.array(critic_reel_cp[1:]) - np.array(critic_reel_cp[:-1])
        #         advantages = discounted_cumulative_sums(
        #             deltas, self.gamma * self.lam
        #         )
        #         adv_mean, adv_std = (
        #             np.mean(advantages),
        #             np.std(advantages)
        #         )
        #         advantages = (advantages - adv_mean) / adv_std
        #
        #         # Compute returns
        #         returns = discounted_cumulative_sums(
        #             reward_reel, self.gamma
        #         )[:-1]
        #
        #         # Tensorize
        #         state_tensor = tf.convert_to_tensor(state_reel[:-1], dtype=tf.float32)
        #         log_prob_tensor = tf.convert_to_tensor(log_prob_reel[:-1], dtype=tf.float32)
        #         action_tensor = tf.convert_to_tensor(action_reel[:-1], dtype=tf.int32)
        #
        #         advantage_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        #         returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        #         # print(state_tensor.shape)      # shape: (500, 3)
        #         # print(advantage_tensor.shape)  # shape: (499)
        #         # print(returns.shape)           # shape: (499)
        #         # exit(0)
        #
        #         # Record
        #         sat_states.append(state_tensor)
        #         sat_log_probs.append(log_prob_tensor)
        #         sat_advantages.append(advantage_tensor)
        #         sat_returns.append(returns_tensor)
        #         sat_actions.append(action_tensor)
        #
        #
        #     # Tensorize
        #     sat_returns_tensor = tf.convert_to_tensor(sat_returns, dtype=tf.float32)
        #
        #
        #     total_advantages.append(sat_advantages)
        #     total_returns.append(sat_returns_tensor)
        #     total_log_probs.append(sat_log_probs)
        #     total_actions.append(sat_actions)
        #     total_states.append(sat_states)
        #
        #     sat_trajectories.append(deepcopy([sat['experience_reels'][r] for r in rand_reels]))
        #     sat_trajectory_critic_values.append(deepcopy([sat['critic_reels'][r] for r in rand_reels]))
        #
        # # shared critic inputs
        # critic_returns = tf.zeros_like(total_returns[0])  # shape: (batch_size, trajectory_len) --> (1, 499)
        # for sat_returns in total_returns:
        #     critic_returns += sat_returns

        # return sat_trajectories, sat_trajectory_critic_values, total_advantages, critic_returns, total_log_probs, total_actions, total_states





