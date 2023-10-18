import numpy as np
import os
import csv
import random
import datetime
import multiprocessing
from functools import partial
from tqdm import tqdm
from mcts_planner import monte_carlo_tree_search
from dp_planner import graph_search, graph_search_events, graph_search_events_interval
from models.SatelliteMLP import SatelliteMLP
from models.SatelliteTransformer import SatelliteTransformer
import tensorflow as tf
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt
import config
from deepdiff import DeepDiff
import pymap3d as pm
import math
import tensorflow_addons as tfa
import threading

from planners.AbstractRL import AbstractRL


# Utility functions
from planners import utils


class TransformerRL(AbstractRL):

    def __init__(self, settings):
        super().__init__(settings)

        self.pool_size = 32  # config.cores
        self.directory = settings["directory"] + "orbit_data/"
        self.settings = settings

        # Hyperparameters
        self.num_epochs = 5
        self.target_update_frequency = 3
        self.replay_batch_size = 32
        self.replay_frequency = 1
        self.buffer_init_size = 50
        self.clip_gradients = False

        # 3. Optimizers
        self.gamma = 0.9

        # 4. Metrics
        self.plot_frequency = 250

    def init_models(self):
        for satellite in self.satellites:
            satellite['q_network'] = SatelliteTransformer().implicit_build()
            satellite['target_q_network'] = SatelliteTransformer().implicit_build()

    # --------------------------------------------
    # Train Planners
    # --------------------------------------------

    def train_planners(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs

        for epoch in range(num_epochs):

            # 1. Plan mission
            self.plan_mission()

            # 2. Record planning results
            self.record_planning_results(epoch)

    def plan_mission(self):

        # 1. Reset all satellite observations, states
        for sat in self.satellites:
            sat['all_obs'] = []
            sat['sat_time'] = 0.0
            sat['sat_angle'] = 0.0
            sat['has_actions'] = True

        # 2. Gather sats that have actions left to take
        counter = 0
        actionable_sats = [sat for sat in self.satellites if sat['has_actions'] is True]
        while(len(actionable_sats) > 0):

            # 3. Take an action for each actionable sat
            for idx, sat in enumerate(actionable_sats):
                debug = False
                if idx == 0 and counter % 10 == 0:
                    debug = True
                init = False
                self.satellite_action(sat, debug, init)  # 0.2 seconds per satellite

            # 4. Update satellite policies
            self.record_satellite_experience()
            if self.steps > self.replay_batch_size and self.steps % self.replay_frequency == 0 and self.steps > self.buffer_init_size:
                self.update_satellite_models()

            # 5. Reset action tracker
            for sat in self.satellites:
                sat['took_action'] = False

            # 6. Regather actionable sats
            actionable_sats = [sat for sat in self.satellites if sat['has_actions'] is True]

            # 7. Copy over q_network to target_q_network
            if counter % self.target_update_frequency == 0:
                for sat in self.satellites:
                    sat['target_q_network'].load_target_weights(sat['q_network'])

            # 8. Plot results
            if counter > 0 and counter % self.plot_frequency == 0:
                self.plot_results(counter)

            # 7. Break if debugging
            counter += 1

            self.steps += 1
            # if counter > 100:
            #     break

    def satellite_action(self, satellite, debug=False, init=False):
        obs_list = satellite['obs_list']
        sat_time = satellite['sat_time']
        sat_angle = satellite['sat_angle']
        last_obs = satellite['last_obs']
        sat_lat = satellite['sat_lat']
        sat_lon = satellite['sat_lon']

        # 1. Get action space (size 10)
        actions = self.get_action_space(sat_time, sat_angle, obs_list, last_obs, self.settings)
        # actions = utils.get_action_space(sat_time, sat_angle, obs_list, last_obs, self.settings)
        num_actions = len(actions)
        if num_actions == 0:
            satellite['has_actions'] = False
            satellite['took_action'] = False
            return

        # 2. Create q_network state, record
        state = deepcopy([sat_time, sat_angle, sat_lat, sat_lon])
        # state = deepcopy([sat_time, sat_angle])
        satellite['last_state'] = state
        # print('--> STATE', state)

        # 3. Get q_network action, record
        q_network = satellite['q_network']
        action_idx = q_network.get_aciton(state, num_actions=num_actions, debug=debug, rand_action=False, init_action=init)
        satellite['last_action_idx'] = action_idx
        # print('--> ACTION SELECTED:', action_idx)

        # 4. Get time distance to last possible action
        last_action = actions[-1]
        last_action_time = last_action["soonest"]

        # 4. Record selected action
        action = actions[action_idx]
        satellite['all_obs'].append(deepcopy(action))
        satellite['location_list'].append(action["location"])  # already seen by this sat
        satellite['last_obs'] = deepcopy(action)
        satellite['sat_time'] = action["soonest"]
        satellite['sat_angle'] = action["angle"]
        satellite['took_action'] = True

        # 5. Record new lat and lon
        sat_time_approx = int(round(action["soonest"] / 10.0, 0))
        nadir_lat_lons = satellite['nadir_lat_lons'][sat_time_approx]
        time, lat, lon = nadir_lat_lons[0], nadir_lat_lons[1], nadir_lat_lons[2]
        satellite['sat_lat'] = lat
        satellite['sat_lon'] = lon

        # 6. Calculate action reward
        reward = action["reward"]
        reward += self.find_event_bonus(action, satellite['sat_time'])
        satellite['rewards'].append(deepcopy(reward))

    def record_satellite_experience(self):

        # 0. Get satellites that took actions
        trainable_sats = [sat for sat in self.satellites if sat['took_action'] is True]

        # 1. Get cumulative reward
        all_rewards = []
        for sat in trainable_sats:
            if sat['took_action'] is True and len(sat['rewards']) > 0:
                all_rewards.append(sat['rewards'][-1])
        cumulative_reward = sum(all_rewards)
        self.step_rewards.append(cumulative_reward)
        self.step_observations.append(len(trainable_sats))

        # 2. (q_network) Find summed q_values for last (actually current) state
        # last_q_values = []
        last_action_indices = []
        for sat in trainable_sats:
            last_state = sat['last_state']
            last_action_idx = sat['last_action_idx']
            last_action_indices.append(last_action_idx)
            # last_q_value = sat['q_network'].get_q_value(last_state, action_idx=last_action_idx)
            # last_q_values.append(last_q_value)

        # 3. (target_q_network) Find summed q_values for "next" state
        # target_q_values = []
        target_states = []
        for sat in trainable_sats:
            curr_state = [sat['sat_time'], sat['sat_angle'], sat['sat_lat'], sat['sat_lon']]
            # curr_state = [sat['sat_time'], sat['sat_angle']]
            target_states.append(curr_state)
            # curr_q_value = sat['target_q_network'].get_q_value(curr_state)
            # target_q_values.append(curr_q_value)

        # 4. Add to experience replay buffer
        for idx, sat in enumerate(trainable_sats):
            sat['q_network'].record_experience(
                sat['last_state'],
                last_action_indices[idx],
                all_rewards[idx],
                target_states[idx]
            )

    def update_satellite_models(self):

        # 0. Get satellites that took actions
        prep_start = time.time()

        def sample_pair(max_len, output_len):
            first_num = random.randint(0, max_len - output_len)
            second_num = random.randint(first_num + 1, min(first_num + output_len, max_len))
            return [first_num, second_num]

        # 0. Get satellites that took actions
        trainable_sats = [sat for sat in self.satellites if sat['took_action'] is True]

        # 1. Sample experiences from experience replay buffer
        min_buf_size = 1e9
        for sat in trainable_sats:
            if len(sat['q_network'].replay_buffer) < min_buf_size:
                min_buf_size = len(sat['q_network'].memory)
        clip_indices = [sample_pair(min_buf_size, config.sequence_len) for x in range(self.replay_batch_size)]
        sat_experience_sequences = []  # (num_sats, batch_size, seq_length, 5)
        for sat in trainable_sats:
            sat_experience_sequences.append(sat['q_network'].sample_memory_trajectory(clip_indices))


        all_rewards = []      # (num_sats, batch_size, seq_length, 1)
        all_states = []       # (num_sats, batch_size, seq_length, 4)
        all_actions = []      # (num_sats, batch_size, seq_length, 1)
        all_next_states = []  # (num_sats, batch_size, seq_length, 4)
        all_next_actions = []
        for experience_sequence in sat_experience_sequences:
            sat_sequence_rewards = []
            sat_sequence_states = []
            sat_sequence_actions = []
            sat_sequence_next_states = []
            sat_sequence_next_actions = []
            for sequence_memory in experience_sequence:
                sequence_rewards = utils.extract_from_batch(sequence_memory, 'reward')
                sequence_states = utils.extract_from_batch(sequence_memory, 'state')
                sequence_actions = utils.extract_from_batch(sequence_memory, 'action')
                sat_next_actions = sequence_actions[1:]
                sequence_next_states = utils.extract_from_batch(sequence_memory, 'next_state')
                while len(sequence_rewards) < config.sequence_len:
                    sequence_rewards.append(0.0)


                sat_sequence_rewards.append(sequence_rewards)
                sat_sequence_states.append(sequence_states)
                sat_sequence_actions.append(sequence_actions)
                sat_sequence_next_actions.append(sat_next_actions)
                sat_sequence_next_states.append(sequence_next_states)

            all_rewards.append(sat_sequence_rewards)
            all_states.append(sat_sequence_states)
            all_actions.append(sat_sequence_actions)
            all_next_states.append(sat_sequence_next_states)
            all_next_actions.append(sat_sequence_next_actions)

        all_actions_copy = deepcopy(all_actions)
        cum_rewards = tf.convert_to_tensor(all_rewards)
        cumulative_reward = tf.reduce_sum(cum_rewards, axis=-1)  # shape: (num_sats, batch_size)
        cumulative_reward = tf.reduce_sum(cumulative_reward, axis=0)  # shape: (batch_size,)




        batch_action_tensors = []
        batch_action_indices = []
        batch_action_mask = []
        batch_state_sequences = []
        batch_next_state_sequences = []

        batch_next_action_tensors = []
        batch_next_action_indices = []
        batch_next_action_mask = []
        for idx1, sat in enumerate(trainable_sats):
            batch_states = all_states[idx1]  # shape: (batch_size, seq_length, 4)b
            batch_actions = all_actions_copy[idx1]  # shape: (batch_size, seq_length, 1)
            batch_next_actions = all_next_actions[idx1]  # shape: (batch_size, seq_length, 1) .. skip first action bc next states

            state_sequence_tensor = sat['q_network'].get_state_sequence_batch(batch_states)
            batch_state_sequences.append(state_sequence_tensor)

            next_state_sequence_tensor = sat['q_network'].get_state_sequence_batch(all_next_states[idx1])
            batch_next_state_sequences.append(next_state_sequence_tensor)


            # Action tensor
            action_tensors = []
            action_mask_tensors = []
            action_idxs_tensors = []
            for idx, batch_element in enumerate(batch_actions):
                input_actions = [str(action) for action in batch_element]
                input_actions.insert(0, '[start]')
                if len(input_actions) > config.sequence_len:
                    input_actions = input_actions[:config.sequence_len]
                input_actions_str = ' '.join(input_actions)
                action_tensor = config.encode(input_actions_str)
                # action_tensor = tf.expand_dims(action_tensor, axis=0)  # shape: (1, sequence_len)
                action_tensors.append(action_tensor)

                sequence_mask = [1] * len(input_actions)
                while len(sequence_mask) < config.sequence_len:
                    sequence_mask.append(0)
                sequence_mask = tf.convert_to_tensor(sequence_mask, dtype=tf.float32)
                # sequence_mask = tf.expand_dims(sequence_mask, axis=0)
                action_mask_tensors.append(sequence_mask)

                # Record and Pad action indices
                while len(batch_actions[idx]) < config.sequence_len:
                    batch_actions[idx].append(0)
                batch_actions[idx] = tf.convert_to_tensor(batch_actions[idx], dtype=tf.int64)
                # batch_actions[idx] = tf.expand_dims(batch_actions[idx], axis=0)
                action_idxs_tensors.append(batch_actions[idx])

            batch_action_tensors.append(tf.convert_to_tensor(action_tensors, dtype=tf.int32))
            batch_action_mask.append(tf.convert_to_tensor(action_mask_tensors, dtype=tf.float32))
            batch_action_indices.append(tf.convert_to_tensor(action_idxs_tensors, dtype=tf.int32))

            # Next Action tensor
            next_action_tensors = []
            next_action_mask_tensors = []
            next_action_idxs_tensors = []
            for idx, batch_element in enumerate(batch_next_actions):
                input_actions = [str(action) for action in batch_element]
                input_actions.insert(0, '[start]')
                if len(input_actions) > config.sequence_len:
                    input_actions = input_actions[:config.sequence_len]
                input_actions_str = ' '.join(input_actions)
                action_tensor = config.encode(input_actions_str)
                # action_tensor = tf.expand_dims(action_tensor, axis=0)  # shape: (1, sequence_len)
                next_action_tensors.append(action_tensor)

                sequence_mask = [1] * len(input_actions)
                while len(sequence_mask) < config.sequence_len:
                    sequence_mask.append(0)
                sequence_mask = tf.convert_to_tensor(sequence_mask, dtype=tf.float32)
                # sequence_mask = tf.expand_dims(sequence_mask, axis=0)
                next_action_mask_tensors.append(sequence_mask)

                # Record and Pad action indices
                while len(batch_next_actions[idx]) < config.sequence_len:
                    batch_next_actions[idx].append(0)
                batch_next_actions[idx] = tf.convert_to_tensor(batch_next_actions[idx], dtype=tf.int64)
                # batch_actions[idx] = tf.expand_dims(batch_actions[idx], axis=0)
                next_action_idxs_tensors.append(batch_next_actions[idx])

            batch_next_action_tensors.append(tf.convert_to_tensor(next_action_tensors, dtype=tf.int32))
            batch_next_action_mask.append(tf.convert_to_tensor(next_action_mask_tensors, dtype=tf.float32))
            batch_next_action_indices.append(tf.convert_to_tensor(next_action_idxs_tensors, dtype=tf.int32))


        # 3.3. (target_q_network) Find summed q_values for "next" state
        target_q_values = []  # shape: (num_sats, batch_size, seq_length)
        for idx, sat in enumerate(trainable_sats):
            # batch_states = all_next_states[idx]  # shape: (batch_size, seq_length, 4)
            # batch_actions = all_next_actions[idx]  # shape: (batch_size, seq_length, 1) .. skip first action bc next states
            # target_q_value = sat['target_q_network'].get_q_value_batch([batch_states, batch_actions])


            target_q_value = sat['target_q_network'].get_q_value_max_batch_fast(
                batch_next_state_sequences[idx],
                batch_next_action_tensors[idx],
                batch_next_action_mask[idx]
            )
            target_q_values.append(target_q_value)  # shape: (batch_size, seq_length)
        summed_q_value_next = tf.reduce_sum(target_q_values,
                                            axis=-1)  # shape: (num_sats, batch_size)  sum sat timestep contribution
        summed_q_value_next = tf.reduce_sum(summed_q_value_next, axis=0)  # shape: (batch_size,)  sum sat contribution
        q_targets = cumulative_reward + self.gamma * summed_q_value_next  # shape: (batch_size,)
        q_targets = tf.cast(q_targets, dtype=tf.float32)
        q_targets_inputs = [q_targets] * len(trainable_sats)

        input_q_values_next = [summed_q_value_next] * len(trainable_sats)

        # 3.2. (q_network) Find summed q_values for last (actually current) state
        last_q_values = []  # shape: (num_sats, batch_size, seq_length)
        last_q_values_sum = []
        for idx, sat in enumerate(trainable_sats):
            # batch_states = deepcopy(all_states[idx])  # shape: (batch_size, seq_length, 4)b
            # batch_actions = deepcopy(all_actions[idx])  # shape: (batch_size, seq_length, 1)
            # last_q_value = sat['q_network'].get_q_value_batch([batch_states, batch_actions], action_idxs=batch_actions)

            last_q_value = sat['q_network'].get_q_value_idx_batch_fast(
                batch_state_sequences[idx],
                batch_action_tensors[idx],
                batch_action_mask[idx],
                batch_action_indices[idx]
            )
            last_q_values.append(last_q_value)
            last_q_values_sum.append(tf.reduce_sum(last_q_value, axis=-1))  # shape: (batch_size,)

        last_q_tensor = tf.stack(last_q_values)  # shape: (num_sats, batch_size, seq_length)
        total_sum = tf.reduce_sum(last_q_tensor, axis=-1)  # shape: (num_sats, batch_size)
        total_sum = tf.reduce_sum(total_sum, axis=0)  # shape: (batch_size,)
        external_sat_sum = []  # shape: (num_sats, batch_size)
        for idx in range(len(trainable_sats)):
            # Subtract the Q-value of the current satellite from the total sum
            sum_without_current_sat = total_sum - last_q_values_sum[idx]  # shape: (batch_size, )
            external_sat_sum.append(
                tf.cast(sum_without_current_sat, dtype=tf.float32)
            )


        # Create Dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            batch_state_sequences,
            batch_action_tensors,
            q_targets_inputs,
            external_sat_sum,
            batch_action_indices,
            batch_action_mask
        ))

        # Train
        curr_time = time.time()
        for idx, data_tuple in enumerate(dataset):
            batch_states, batch_actions, target_q_value, current_q_values, batch_action_indices, batch_action_mask = data_tuple
            trainable_sats[idx]['q_network'].train_step(batch_states, batch_actions, target_q_value, current_q_values, batch_action_indices, batch_action_mask)
        # print('--> PREP / TRAINING TIME:', time.time() - prep_start, time.time() - curr_time)













