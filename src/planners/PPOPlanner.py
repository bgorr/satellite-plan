import config
from planners.AbstractPlanner import AbstractPlanner
from copy import deepcopy
from planners import utils
from models.SatelliteMLP import SatelliteMLP
import random
import tensorflow as tf
import numpy as np
from sampling.PPOSampling import PPOSampling
from models.PlanningCritic import PlanningCritic
import config
import scipy.signal
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOPlanner(AbstractPlanner):

    def __init__(self, settings):
        super().__init__(settings)
        self.train_policy_iterations = config.ppo_train_policy_iterations
        self.train_critic_iterations = config.ppo_train_value_iterations
        self.target_update_frequency = 10
        self.buffer_init_size = config.ppo_buffer_init_size
        self.mini_batch_size = config.ppo_batch_size
        self.episode_steps = config.ppo_episode_steps
        self.critic = PlanningCritic().implicit_build()

    def init_models(self):
        for satellite in self.satellites:
            satellite['q_network'] = SatelliteMLP().implicit_build()
            satellite['target_q_network'] = SatelliteMLP().implicit_build()

    def get_satellite_state(self, sat):
        return [
            # sat['sat_step'],  # norm by 8640
            # sat['sat_obs_idx'],  # norm by 10000
            sat['sat_time'],  # norm by 8640
            sat['sat_angle'],
            # len(sat['storage']),
            # sat['sat_lat'],
            # sat['sat_lon'],
        ]

    # ---------------------------------------------
    # Reset
    # ---------------------------------------------

    def reset_mini_batch(self):
        for sat in self.satellites:
            sat['experience_reels'] = []
            sat['critic_reels'] = []



    # ---------------------------------------------
    # Train episode
    # ---------------------------------------------

    def train_episode(self):
        self.reset_mini_batch()

        for mb in range(self.mini_batch_size):

            self.reset_episode()

            actionable_sats = [sat for sat in self.satellites if sat['has_actions'] is True]
            while (len(actionable_sats) > 0):

                # 1. Take a step in environment, run critic
                if self.sim_step(actionable_sats) is False:
                    break
                self.run_critic()  # All sats 'critic_values' are updated (same for all sats)

                # 2. Reset actions
                for sat in self.satellites:
                    sat['took_action'] = False

                # 3. Copy of target network (VDNs / Q-Learning)
                # if self.step % self.target_update_frequency == 0:
                #     for sat in self.satellites:
                #         if sat['target_q_network']:
                #             sat['target_q_network'].load_target_weights(sat['q_network'])

                # 4. Plot progress
                # if self.step > 0 and self.step % self.plot_frequency == 0:
                #     self.plot_progress()

                # 5. Determine actionable satellites
                actionable_sats = [sat for sat in self.satellites if sat['has_actions'] is True]

                # 6. Increment counters
                if len(actionable_sats) != 0:
                    self.record_step()

                # if self.step > 50:
                #     break


            # if mb == self.mini_batch_size - 1:
            #     self.record_episode()
            #     self.plot_history()

            self.record_episode()


            self.episode += 1

        self.update_satellite_models_ppo()
        self.plot_history()

        return self.satellites

    def run_critic(self):
        all_sat_states = []
        for sat in self.satellites:
            sat_state = sat['experience_buffer'][-1][0]
            all_sat_states.extend(sat_state)
        all_sat_states = tf.convert_to_tensor(all_sat_states, dtype=tf.float32)
        all_sat_states = tf.expand_dims(all_sat_states, axis=0)
        value_t = self.critic(all_sat_states)
        for sat in self.satellites:
            sat['critic_values'].append(value_t.numpy()[0][0])

    # ---------------------------------------------
    # PPO action
    # ---------------------------------------------

    def select_action(self, sat, state, num_actions, rand_action=False):
        if rand_action is True:
            return random.randint(0, num_actions - 1)
        else:
            return sat['q_network'].get_ppo_action(state, num_actions=num_actions)

    def update_satellite_models(self):
        num_reels = min([len(sat['experience_reels']) for sat in self.satellites])
        if num_reels < self.buffer_init_size:
            return

        # 1. Get a mini-batch of trajectories + buffers
        mini_batch, critic_values, advantages, critic_returns, log_probs, actions, states = PPOSampling(self.satellites).sample()  # (num_sats, batch_size, trajectory_len)

        # 2. Train Policy
        for _ in range(self.train_policy_iterations):
            losses = []
            kl_losses = []
            for idx, sat in enumerate(self.satellites):
                sat_states = states[idx]
                sat_actions = actions[idx]
                sat_log_probs = log_probs[idx]
                sat_advantages = advantages[idx]
                sat_losses = []
                sat_kl_losses = []
                for state, action, log_prob, advantage in zip(sat_states, sat_actions, sat_log_probs, sat_advantages):
                    kl, loss, entropy = sat['q_network'].train_step_ppo(state, action, log_prob, advantage)
                    sat_losses.append(loss)
                    sat_kl_losses.append(kl)

                    # if kl > 1.5 * config.ppo_target_kl:
                    if kl > 2.0:
                        print('Early stopping...')
                        break

                losses.append(np.mean(sat_losses))
                kl_losses.append(np.mean(sat_kl_losses))
            print('--> ACTOR LOSS', round(np.mean(losses), 3), '| KL DIVERGENCE', round(np.mean(kl_losses), 3))

        # 3. Train Critic
        critic_inputs = []  # (batch_size, trajectory_len, num_agents * state_dim) --> (1, 499, 3)
        for batch_item_idx in range(self.mini_batch_size):
            batch_item_inputs = []  # (trajectory_len, num_agents * state_dim)
            for step_idx in range(config.ppo_episode_steps-1):
                combined_states = []  # (num_agents * state_dim)
                for sat_idx, sat in enumerate(self.satellites):
                    sat_state = states[sat_idx][batch_item_idx][step_idx]  # (trajectory_len, state_dim)
                    combined_states.extend(sat_state)
                batch_item_inputs.append(combined_states)
            critic_inputs.append(tf.convert_to_tensor(batch_item_inputs, dtype=tf.float32))

        for _ in range(self.train_critic_iterations):  # (batch_size, )
            losses = []
            for batch_item_idx in range(self.mini_batch_size):
                critic_return = critic_returns[batch_item_idx]
                critic_input = critic_inputs[batch_item_idx]
                loss = self.critic.train_step(critic_input, critic_return)
                losses.append(loss)
            print('--> CRITIC LOSS:', round(np.mean(losses), 3))


    def update_satellite_models_ppo(self):

        # 1. Get a mini-batch of trajectories + buffers
        states, actions, log_probs, advantages, critic_inputs, critic_labels = PPOSampling(
            self.satellites).sample()  # (num_sats, batch_size, trajectory_len)

        # 2. Train Policy
        for idx, sat in enumerate(self.satellites):
            sat_states = []
            sat_actions = []
            sat_log_probs = []

            # Iterate over trajectories to flatten
            for state, action, log_prob in zip(states[idx], actions[idx], log_probs[idx]):
                sat_states.extend(state)
                sat_actions.extend(action)
                sat_log_probs.extend(log_prob)

            sat_states = tf.convert_to_tensor(sat_states, dtype=tf.float32)
            sat_actions = tf.convert_to_tensor(sat_actions, dtype=tf.int32)
            sat_log_probs = tf.convert_to_tensor(sat_log_probs, dtype=tf.float32)

            kl = -1
            train_its = 0
            for _ in range(config.ppo_train_policy_iterations):
                train_its += 1
                kl, loss, entropy = sat['q_network'].train_step_ppo(sat_states, sat_actions, sat_log_probs, advantages)
                if kl > 1.5 * config.ppo_target_kl:
                    # Early Stopping
                    break
            print('Sat', idx, '| KL Divergence:', kl.numpy(), '| Loss:', loss.numpy(), '| Entropy:', entropy.numpy(), '| Train Its:', train_its)

        # 3. Train Critic
        critic_loss = -1
        for _ in range(config.ppo_train_value_iterations):
            critic_loss = self.critic.train_step(critic_inputs, critic_labels)
        print('--> CRITIC LOSS:', critic_loss.numpy())


    # ---------------------------------------------
    # Plotting
    # ---------------------------------------------

    def plot_history(self):
        save_path = os.path.join(config.plots_dir, self.settings['name'], self.settings['planner'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if len(self.training_history) > 0:
            history = self.training_history[-1]
            epoch = len(self.training_history)
            self.plot_progress(history=history, epoch=epoch)

        # Episode wise plots
        if len(self.training_history) > 1:
            plot_path = os.path.join(save_path, self.settings['planner'] + '-results.png')
            epochs = [epoch+1 for epoch in range(len(self.training_history))]
            total_obs = [history['total_observations'] for history in self.training_history]
            total_mem_overflows = [history['total_mem_overflows'] for history in self.training_history]
            total_events = [history['total_events_seen'] for history in self.training_history]
            total_unique_events = [history['total_unique_events_seen'] for history in self.training_history]
            total_infeasibilities = [history['total_infeasibilities'] for history in self.training_history]
            total_dl_events = [history['total_dl_events'] for history in self.training_history]
            total_points_seen = [history['total_points_seen'] for history in self.training_history]
            total_actions_taken = [history['actions_taken'] for history in self.training_history]

            # Create a GridSpec object
            gs = gridspec.GridSpec(4, 2)

            fig = plt.figure(figsize=(8, 9))  # default [6.4, 4.8], W x H
            fig.suptitle(self.settings['name'] + ' | ' + self.settings['planner'], fontsize=16)
            # plt.subplot(4, 2, 1)
            plt.subplot(gs[0, 0])
            plt.plot(epochs, total_obs)
            plt.title("Total Observations")
            plt.xlabel('Epoch')
            plt.ylabel('Observations')
            # for epoch in range(0, max(epochs), 10):
            #     plt.axvline(x=epoch, color='r', linestyle='--')

            # plt.subplot(4, 2, 2)
            plt.subplot(gs[0, 1])
            plt.plot(epochs, total_unique_events)  # Change to total unique events for paper
            plt.title("Total Events")
            plt.xlabel('Epoch')
            plt.ylabel('Events')
            # for epoch in range(0, max(epochs), 10):
            #     plt.axvline(x=epoch, color='r', linestyle='--')

            # # plt.subplot(4, 2, 3)
            # plt.subplot(gs[1, 0])
            # plt.plot(epochs, total_unique_events)
            # plt.title("Unique Events")
            # plt.xlabel('Epoch')
            # plt.ylabel('Events')
            window_size = config.ppo_batch_size
            n = len(self.reward_history)
            averaged_rewards = [np.mean(self.reward_history[i:i+window_size]) for i in range(0, n, window_size)]
            adjusted_epochs = range(len(averaged_rewards))

            plt.subplot(gs[1, 0])
            # plt.plot(epochs, self.reward_history)
            plt.plot(adjusted_epochs, averaged_rewards)
            plt.title("Reward Graph")
            plt.xlabel('Epoch')
            plt.ylabel('Reward')
            # for epoch in range(0, max(epochs), 10):
            #     plt.axvline(x=epoch, color='r', linestyle='--')


            # plt.subplot(4, 2, 4)
            plt.subplot(gs[1, 1])
            plt.plot(epochs, total_infeasibilities)
            plt.title("Infeasible Points")
            plt.xlabel('Epoch')
            plt.ylabel('Points')
            # for epoch in range(0, max(epochs), 10):
            #     plt.axvline(x=epoch, color='r', linestyle='--')

            # plt.subplot(gs[1, 1])
            # plt.plot(epochs, total_dl_events)
            # plt.title("Downlinked Events")
            # plt.xlabel('Epoch')
            # plt.ylabel('Events')


            # plt.subplot(4, 2, 5)
            plt.subplot(gs[2, 0])
            plt.plot(epochs, total_points_seen)
            plt.title("Total Points")
            plt.xlabel('Epoch')
            plt.ylabel('Points')
            # for epoch in range(0, max(epochs), 10):
            #     plt.axvline(x=epoch, color='r', linestyle='--')


            # memory overflow actions taken
            # plt.subplot(4, 2, 6)
            plt.subplot(gs[2, 1])
            plt.plot(epochs, total_mem_overflows)
            plt.title("Total Memory Overflows")
            plt.xlabel('Epoch')
            plt.ylabel('MEM Overflows')
            plt.axhline(y=0, color='r', linestyle='--')

            # histogram of actions taken
            # plt.subplot(4, 2, 7)
            plt.subplot(gs[3, :])
            unique_actions = np.unique(np.concatenate(total_actions_taken))
            frequency_matrix = np.zeros((len(total_actions_taken), len(unique_actions)))
            for i, actions in enumerate(total_actions_taken):
                for action in actions:
                    j = np.where(unique_actions == action)[0][0]
                    frequency_matrix[i, j] += 1
            bar_width = 0.025
            index = np.arange(len(unique_actions))
            for i, frequency in enumerate(frequency_matrix):
                plt.bar(index + i * bar_width, frequency, width=bar_width, label=f"List {i + 1}")
            plt.title("Actions Taken")
            plt.xlabel("Action")
            plt.ylabel("Frequency")
            plt.xticks(index + bar_width, unique_actions)




            plt.tight_layout()
            plt.savefig(plot_path)
            plt.show()













