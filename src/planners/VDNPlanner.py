from planners.AbstractPlanner import AbstractPlanner
from copy import deepcopy
from planners import utils
from models.SatelliteMLP import SatelliteMLP
import random
import tensorflow as tf
from sampling.VDNSampling import VDNSampling
import numpy as np



class VDNPlanner(AbstractPlanner):

    def __init__(self, settings):
        super().__init__(settings)
        self.models_update_frequency = 2
        self.target_update_frequency = 10
        self.buffer_init_size = 256
        self.replay_batch_size = 128
        self.gamma = 0.99

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
    # Train episode
    # ---------------------------------------------

    def train_episode(self):
        self.reset_episode()

        actionable_sats = [sat for sat in self.satellites if sat['has_actions'] is True]
        train_steps = 0
        while (len(actionable_sats) > 0):

            # 1. Take a step in environment, run critic
            if self.sim_step(actionable_sats) is False:
                break

            # # 2. Reset actions
            for sat in self.satellites:
                sat['took_action'] = False

            # 5. Determine actionable satellites
            actionable_sats = [sat for sat in self.satellites if sat['has_actions'] is True]

            # 6. Increment counters
            if len(actionable_sats) != 0:
                self.record_step()


            # 7. Update models
            self.update_satellite_models()
            if train_steps % self.target_update_frequency == 0:
                for sat in self.satellites:
                    if sat['target_q_network']:
                        sat['target_q_network'].load_target_weights(sat['q_network'])
            train_steps += 1

            # if self.step > 50:
            #     break


        # if self.episode > 0:
        #     for idx in range(100):
        #         self.update_satellite_models()
        #         if idx % self.target_update_frequency == 0:
        #             for sat in self.satellites:
        #                 if sat['target_q_network']:
        #                     sat['target_q_network'].load_target_weights(sat['q_network'])

        self.record_episode()
        self.plot_history()
        self.episode += 1
        return self.satellites


    # ---------------------------------------------
    # VDN action
    # ---------------------------------------------

    def select_action(self, sat, state, num_actions, rand_action=False):
        if rand_action is True:
            return random.randint(0, num_actions - 1)
        else:
            return sat['q_network'].get_action(state, num_actions=num_actions, debug=False)

    def update_satellite_models(self):
        trainable_sats = self.satellites
        sat_experiences = VDNSampling(self.satellites, self.replay_batch_size).sample()
        batch_states = [  # (num_sats, batch_size, state_dim)
            tf.convert_to_tensor(utils.idx_from_batch(experience, 0), dtype=tf.float32) for experience in
            sat_experiences
        ]
        batch_actions = [  # (num_sats, batch_size, action_dim)
            tf.convert_to_tensor(utils.idx_from_batch(experience, 1), dtype=tf.int32) for experience in
            sat_experiences
        ]
        batch_rewards = [  # (num_sats, batch_size)
            tf.convert_to_tensor(utils.idx_from_batch(experience, 2), dtype=tf.float32) for experience in
            sat_experiences
        ]
        batch_next_states = [  # (num_sats, batch_size, state_dim)
            tf.convert_to_tensor(utils.idx_from_batch(experience, 3), dtype=tf.float32) for experience in
            sat_experiences
        ]
        # Cumulative reward across sat batches
        cumulative_reward = tf.reduce_sum(batch_rewards, axis=0)  # shape: (batch_size,)

        # Find target q_values for next state
        target_q_values = []  # shape: (num_sats, batch_size)
        sat_time_differences = []

        for idx, sat in enumerate(trainable_sats):
            # INPUT: next_states (batch_size, 2)
            # OUTPUT: target_q_values (batch_size,)
            b_states = deepcopy(batch_states[idx])  # shape: (batch_size, state_dim)
            b_next_states = deepcopy(batch_next_states[idx])  # shape: (batch_size, state_dim)
            time_diff = b_next_states[:, 0] - b_states[:, 0]  # shape: (batch_size,)
            sat_time_differences.append(time_diff)
            target_q_value_idx = sat['q_network'].get_q_idx_batch_max(batch_next_states[idx])
            target_q_value = sat['target_q_network'].get_q_value_batch_idx(batch_next_states[idx], action_idxs=target_q_value_idx)
            # target_q_value = sat['target_q_network'].get_q_value_batch_max(batch_next_states[idx])
            target_q_values.append(target_q_value)

        sat_time_differences_tensor = tf.stack(sat_time_differences)  # shape: (num_sats, batch_size)
        target_q_tensor = tf.stack(target_q_values)  # shape: (num_sats, batch_size)

        # Lambda rate parameter
        lambda_rate = tf.constant(0.01, dtype=tf.float32)  # You can tune this value

        # Calculate the exponential discount factor
        exp_discount_factor = tf.exp(-lambda_rate * sat_time_differences_tensor)
        # exp_discount_factor = 1.0

        # Compute the discounted Q-values
        discounted_q_values = target_q_tensor * exp_discount_factor  # shape: (num_sats, batch_size)
        discounted_q_values = tf.reduce_sum(discounted_q_values, axis=0)  # shape: (batch_size,)

        summed_q_value_next = tf.reduce_sum(target_q_tensor, axis=0)  # shape: (batch_size,)

        # target_q_value = cumulative_reward + self.gamma * summed_q_value_next  # shape: (batch_size,)
        target_q_value = cumulative_reward + discounted_q_values  # shape: (batch_size,)


        target_q_value = tf.cast(target_q_value, dtype=tf.float32)
        target_q_value_inputs = [target_q_value] * len(trainable_sats)

        # Find summed q_values for last (actually current) state
        last_q_values = []  # shape: (num_sats, batch_size)
        for idx, sat in enumerate(trainable_sats):
            # INPUT: states (batch_size, 2), actions (batch_size,)
            # OUTPUT: last_q_value (batch_size,)
            batch_state = batch_states[idx]
            batch_action = batch_actions[idx]
            last_q_value = sat['q_network'].get_q_value_batch(batch_state, action_idxs=batch_action)
            last_q_values.append(last_q_value)
        last_q_tensor = tf.stack(last_q_values)  # shape: (num_sats, batch_size)
        total_sum = tf.reduce_sum(last_q_tensor, axis=0)  # shape: (batch_size,)
        external_sat_sum = []
        for idx in range(len(trainable_sats)):
            # Subtract the Q-value of the current satellite from the total sum
            sum_without_current_sat = total_sum - last_q_values[idx]  # shape: (batch_size,)
            external_sat_sum.append(
                tf.cast(sum_without_current_sat, dtype=tf.float32)
            )

        # Create Dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (batch_states, batch_actions, target_q_value_inputs, external_sat_sum))

        # Train
        losses = []
        for idx, data_tuple in enumerate(dataset):
            batch_states, batch_actions, target_q_value, current_q_values = data_tuple
            metrics = trainable_sats[idx]['q_network'].train_step(batch_states, batch_actions, target_q_value,
                                                                  current_q_values)
            losses.append(metrics['loss'])

        if self.satellites[0]['q_network'].step % 1000 == 0:
            print('--> AVERAGE LOSS:', round(np.mean(losses), 3), '| EPSILON:', round(np.mean([sat['q_network'].linear_decay() for sat in self.satellites]), 2), '| STEP:', round(np.mean([sat['q_network'].step for sat in self.satellites]), 2))

















