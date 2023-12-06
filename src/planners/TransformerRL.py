import numpy as np
import os
import csv
import random
import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
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
        self.init_done = False
        self.directory = settings["directory"] + "orbit_data/"
        self.settings = settings

        # Hyperparameters
        self.num_epochs = 3
        self.target_update_frequency = 2
        self.replay_batch_size = 16
        self.replay_frequency = 1
        self.buffer_init_size = 200
        self.clip_gradients = False


        # 3. Optimizers
        self.gamma = 0.9

        # 4. Metrics
        self.plot_frequency = 50

        # 5. Executors
        self.executor = ProcessPoolExecutor(max_workers=12)

    def init_models(self):
        for satellite in self.satellites:
            satellite['q_network'] = SatelliteTransformer().implicit_build()
            satellite['target_q_network'] = SatelliteTransformer().implicit_build()

    # --------------------------------------------
    # Train Planners
    # --------------------------------------------

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
        while (len(actionable_sats) > 0):

            # 3. Take asynchronous actions
            took_action = self.async_step(actionable_sats, counter)
            if took_action is False:
                break


            if counter % 10 == 0:
                sat_steps = [round(sat['q_network'].step / (counter+2), 2) for sat in self.satellites]
                sat_steps_avg = round(sum(sat_steps) / len(sat_steps), 2)
                print('UPDATES', counter, ' AVG | ACTIONS :',  sat_steps_avg)


            # 4. Update satellite DQNs
            min_memory_len = min([len(sat['q_network'].memory) for sat in self.satellites])
            if min_memory_len > self.buffer_init_size:
                self.init_done = True
                self.update_satellite_models()

            # if self.steps > self.replay_batch_size and self.steps % self.replay_frequency == 0 and self.steps > self.buffer_init_size:
            #     self.update_satellite_models()

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

            # 9. Break / debugging
            counter += 1
            self.steps += 1
            self.step_events.append(0)
            self.step_observations.append(0)

            sats_obs_left = []
            sats_timestamps = []
            for sat in self.satellites:
                sats_obs_left.append(deepcopy(sat['obs_left']))
                sats_timestamps.append(deepcopy(sat['sat_time']))
            self.sat_observations_left.append(sats_obs_left)
            self.sat_timesteps.append(sats_timestamps)
            self.init_done = False

            # if counter > 100:
            #     break

    def satellite_action(self, satellite, debug=False):
        obs_list = satellite['obs_list']
        sat_time = satellite['sat_time']
        sat_angle = satellite['sat_angle']
        last_obs = satellite['last_obs']
        sat_lat = satellite['sat_lat']
        sat_lon = satellite['sat_lon']

        # 1. Get action space (size 10)
        actions, obs_left = self.get_action_space(sat_time, sat_angle, obs_list, last_obs, self.settings)
        num_actions = len(actions)
        if num_actions == 0:
            satellite['has_actions'] = False
            satellite['took_action'] = False
            return False
        satellite['obs_left'] = obs_left

        # 2. Create q_network state, record
        state = deepcopy([sat_time, sat_angle, sat_lat, sat_lon])
        # state = deepcopy([sat_time, sat_angle])
        satellite['last_state'] = state

        # 3. Get q_network action, record
        q_network = satellite['q_network']
        action_idx = q_network.get_action(state, num_actions=num_actions, debug=debug, rand_action=False)
        satellite['last_action_idx'] = action_idx

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
        sat_time_approx = int(action["soonest"])
        nadir_lat_lons = satellite['nadir_lat_lons'][sat_time_approx]
        time, lat, lon = nadir_lat_lons[0], nadir_lat_lons[1], nadir_lat_lons[2]
        satellite['sat_lat'] = lat
        satellite['sat_lon'] = lon

        # 6. Find event bonus
        reward = action["reward"]
        event_bonus = self.find_event_bonus(action, satellite)
        reward += event_bonus

        # 7. Find event clock penalty
        if event_bonus > 0.0:
            satellite['last_event_time'] = satellite['sat_time']
            self.step_events[self.steps] += 1
        # time_since_last_event = satellite['sat_time'] - satellite['last_event_time']
        # tsle_penalty = 1.0 * (time_since_last_event / 500.0)
        # if debug is True:
        #     print('--> TIME SINCE LAST EVENT:', time_since_last_event, tsle_penalty)
        # reward -= tsle_penalty
        # reward = float(reward)

        # 8. Add go experience replay buffer
        satellite['q_network'].record_experience(
            satellite['last_state'],
            satellite['last_action_idx'],
            reward,
            [satellite['sat_time'], satellite['sat_angle'], satellite['sat_lat'], satellite['sat_lon']]
        )

        # 8. Record reward
        satellite['rewards'].append(deepcopy(reward))
        return True








    def update_satellite_models(self):

        # 0. Get satellites that took actions
        prep_start = time.time()

        def sample_pair(max_len, output_len):
            first_num = random.randint(0, max_len - output_len)
            second_num = random.randint(first_num + 1, min(first_num + output_len, max_len))
            return [first_num, second_num]

        # 0. Get satellites that took actions
        trainable_sats = [sat for sat in self.satellites if sat['took_action'] is True]

        # --------------------------------
        # 1. New sampling method
        # --------------------------------

        sat_experience_sequences = self.sample_satellite_time_windows(trainable_sats)

        # --------------------------------
        # 2. Cumulative reward
        # --------------------------------

        # Initialize lists
        all_rewards, all_states, all_actions, all_next_states, all_next_actions = [], [], [], [], []

        # Loop through satellite experience sequences
        for experience_sequence in sat_experience_sequences:
            sat_sequence_states = TransformerRL.extract_elements_from_sequences(experience_sequence, 'state', config)
            sat_sequence_actions = TransformerRL.extract_elements_from_sequences(experience_sequence, 'action', config)
            sat_sequence_rewards = TransformerRL.extract_elements_from_sequences(experience_sequence, 'reward', config)
            sat_sequence_next_states = TransformerRL.extract_elements_from_sequences(experience_sequence, 'next_state', config)

            # Pad rewards
            for sequence_rewards, sequence_actions in zip(sat_sequence_rewards, sat_sequence_actions):
                while len(sequence_rewards) < config.sequence_len:
                    sequence_rewards.append(0.0)
                while len(sequence_actions) < config.sequence_len:
                    sequence_actions.append(0)

            # Remove first action from next actions
            sat_sequence_next_actions = [a[1:] for a in sat_sequence_actions]

            all_rewards.append(sat_sequence_rewards)
            all_states.append(sat_sequence_states)
            all_actions.append(sat_sequence_actions)
            all_next_states.append(sat_sequence_next_states)
            all_next_actions.append(sat_sequence_next_actions)

        # Convert to tensors and compute cumulative rewards
        cum_rewards = tf.convert_to_tensor(all_rewards)
        cumulative_reward = tf.reduce_sum(cum_rewards, axis=-1)
        cumulative_reward = tf.reduce_sum(cumulative_reward, axis=0)

        # --------------------------------
        # Next Actions
        # --------------------------------

        sat_data_list = []
        for idx1, sat in enumerate(trainable_sats):
            sat_data = {
                'all_states': all_states[idx1],
                'all_next_states': all_next_states[idx1],
                'all_actions': all_actions[idx1],
                'all_next_actions': all_next_actions[idx1],
                'state_vars': sat['q_network'].state_vars
            }
            sat_data_list.append(sat_data)

        # Initialize result lists
        batch_state_sequences, batch_next_state_sequences = [], []
        batch_action_tensors, batch_action_mask, batch_action_indices = [], [], []
        batch_next_action_tensors, batch_next_action_mask = [], []
        batch_state_mask, batch_next_state_mask = [], []

        # Execute in parallel
        results = list(self.executor.map(TransformerRL.process_actions_proc, sat_data_list))

        for result in results:
            batch_state_sequences.append(result['batch_state_sequences'])
            batch_next_state_sequences.append(result['batch_next_state_sequences'])
            batch_action_tensors.append(result['batch_action_tensors'])
            batch_action_mask.append(result['batch_action_mask'])
            batch_action_indices.append(result['batch_action_indices'])
            batch_next_action_tensors.append(result['batch_next_action_tensors'])
            batch_next_action_mask.append(result['batch_next_action_mask'])
            batch_state_mask.append(result['batch_state_masks'])
            batch_next_state_mask.append(result['batch_next_state_masks'])

        # --------------------------------
        # Q Values
        # --------------------------------

        curr_time = time.time()
        # 3.3. (target_q_network) Find summed q_values for "next" state
        target_q_values = []  # shape: (num_sats, batch_size, seq_length)
        for idx, sat in enumerate(trainable_sats):
            target_q_value = sat['target_q_network'].get_q_value_max_batch_fast(
                batch_next_state_sequences[idx],
                batch_next_action_tensors[idx],
                batch_next_action_mask[idx],
                batch_next_state_mask[idx],
            )
            target_q_values.append(target_q_value)  # shape: (batch_size, seq_length)
        summed_q_value_next = tf.reduce_sum(target_q_values, axis=-1)  # shape: (num_sats, batch_size)  sum sat timestep contribution
        summed_q_value_next = tf.reduce_sum(summed_q_value_next, axis=0)  # shape: (batch_size,)  sum sat contribution
        q_targets = cumulative_reward + self.gamma * summed_q_value_next  # shape: (batch_size,)
        q_targets = tf.cast(q_targets, dtype=tf.float32)
        q_targets_inputs = [q_targets] * len(trainable_sats)
        # print('TARGET Q VALUE TIME', time.time() - curr_time)



        # 3.2. (q_network) Find summed q_values for last (actually current) state
        curr_time = time.time()
        last_q_values = []  # shape: (num_sats, batch_size, seq_length)
        last_q_values_sum = []
        for idx, sat in enumerate(trainable_sats):
            last_q_value = sat['q_network'].get_q_value_idx_batch_fast(
                batch_state_sequences[idx],
                batch_action_tensors[idx],
                batch_action_mask[idx],
                batch_action_indices[idx],
                batch_state_mask[idx]
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
        # print('LAST Q VALUE TIME', time.time() - curr_time)

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
        losses = []
        for idx, data_tuple in enumerate(dataset):
            batch_states, batch_actions, target_q_value, current_q_values, batch_action_indices, batch_action_mask = data_tuple
            metrics = trainable_sats[idx]['q_network'].train_step(batch_states, batch_actions, target_q_value, current_q_values, batch_action_indices, batch_action_mask)
            losses.append(metrics['loss'])
        # print('--> PREP / TRAINING TIME:', time.time() - prep_start, time.time() - curr_time)
        print('--> AVERAGE LOSS:', np.mean(losses))





    def sample_satellite_time_windows(self, trainable_sats):

        def sample_time_window(ragged_memory_matrix, lower_bound_max, upper_bound_min):
            t_start = random.uniform(lower_bound_max, upper_bound_min)
            end_times = []
            for sat_memory_seq in ragged_memory_matrix:
                sat_end_times = []
                for memory_time in sat_memory_seq:
                    if t_start > memory_time:
                        continue
                    sat_end_times.append(memory_time)
                    if len(sat_end_times) >= config.sequence_len:
                        break
                if len(sat_end_times) < config.sequence_len:
                    return None
                end_times.append(sat_end_times)

            # assumptions, 10 experience replay steps exist for each sat
            # find idx of sat whose end time is smallest, this will be the sat that keeps all of its experience
            min_idx = np.argmin([x[-1] for x in end_times])
            min_end_time = end_times[min_idx][-1]
            pruned_end_times = []
            for idx, action_times in enumerate(end_times):
                if idx == min_idx:
                    pruned_end_times.append(action_times)
                else:
                    pruned_items = [time_e for time_e in action_times if time_e <= min_end_time]
                    if len(pruned_items) == 0:
                        return None
                    pruned_end_times.append(pruned_items)
            return pruned_end_times

        def sample_time_window_batch(ragged_memory_matrix, batch_size):
            upper_bound_min = min([x[-1] for x in ragged_memory_matrix])
            lower_bound_max = max([x[0] for x in ragged_memory_matrix])

            # 1. Pick random start time
            # 2. Find all end times for each satellite
            batch = []
            while(len(batch) < batch_size):
                window = sample_time_window(ragged_memory_matrix, lower_bound_max, upper_bound_min)
                counter = 0
                while window is None:
                    window = sample_time_window(ragged_memory_matrix, lower_bound_max, upper_bound_min)
                    counter += 1
                    if counter > 100:
                        raise ValueError("--> COULDNT SAMPLE TIME WINDOW")
                batch.append(window)
            return batch

        memory_matrix = [sat['q_network'].get_memory_time_values() for sat in trainable_sats]
        memory_batch = sample_time_window_batch(memory_matrix, self.replay_batch_size)
        clip_indices = []
        for batch in memory_batch:
            for idx, sat_memory in enumerate(batch):
                if len(clip_indices) <= idx:
                    clip_indices.append([])
                action_start_idx = memory_matrix[idx].index(sat_memory[0])
                action_end_idx = memory_matrix[idx].index(sat_memory[-1])
                clip_indices[idx].append([action_start_idx, action_end_idx])

        sat_experience_sequences = []  # (num_sats, batch_size, seq_length, 5)
        for idx, sat in enumerate(trainable_sats):
            sat_experience_sequences.append(sat['q_network'].sample_memory_trajectory(clip_indices[idx]))

        return sat_experience_sequences

    @staticmethod
    def process_actions_proc(sat_data):

        batch_states = sat_data['all_states']
        batch_next_states = sat_data['all_next_states']
        batch_actions = sat_data['all_actions']
        batch_next_actions = sat_data['all_next_actions']
        state_vars = sat_data['state_vars']

        # Padding states
        batch_states_mask = []
        for states in batch_states:
            mask = [1] * len(states)
            while len(states) < config.sequence_len:
                states.append([0] * state_vars)
                mask.append(0)
            batch_states_mask.append(mask)

        # Padding next states
        batch_next_states_mask = []
        for states in batch_next_states:
            mask = [1] * len(states)
            while len(states) < config.sequence_len:
                states.append([0] * state_vars)
                mask.append(0)
            batch_next_states_mask.append(mask)

        # Convert to tensors
        state_sequence_tensor = tf.convert_to_tensor(batch_states, dtype=tf.float32)
        next_state_sequence_tensor = tf.convert_to_tensor(batch_next_states, dtype=tf.float32)
        
        state_mask_tensor = tf.convert_to_tensor(batch_states_mask, dtype=tf.float32)
        next_state_mask_tensor = tf.convert_to_tensor(batch_next_states_mask, dtype=tf.float32)

        # Process actions
        action_tensors, action_mask_tensors, action_idxs_tensors = TransformerRL.process_batch_actions(batch_actions)
        next_action_tensors, next_action_mask_tensors, next_action_idxs_tensors = TransformerRL.process_batch_actions(batch_next_actions)

        return {
            'batch_state_sequences': state_sequence_tensor,
            'batch_next_state_sequences': next_state_sequence_tensor,
            'batch_action_tensors': action_tensors,
            'batch_action_mask': action_mask_tensors,
            'batch_action_indices': action_idxs_tensors,
            'batch_next_action_tensors': next_action_tensors,
            'batch_next_action_mask': next_action_mask_tensors,
            'batch_state_masks': state_mask_tensor,
            'batch_next_state_masks': next_state_mask_tensor
        }

    @staticmethod
    def process_batch_actions(batch_actions):
        action_tensors, action_mask_tensors, action_idxs_tensors = [], [], []
        for batch_element in batch_actions:
            input_actions = ['[start]'] + [str(action) for action in batch_element]
            if len(input_actions) > config.sequence_len:
                input_actions = input_actions[:config.sequence_len]
            input_actions_str = ' '.join(input_actions)

            action_tensor = config.encode(input_actions_str)
            action_tensors.append(action_tensor)

            sequence_mask = [1] * len(input_actions) + [0] * (config.sequence_len - len(input_actions))
            action_mask_tensors.append(tf.convert_to_tensor(sequence_mask, dtype=tf.float32))

            padded_actions = batch_element + [0] * (config.sequence_len - len(batch_element))
            action_idxs_tensors.append(tf.convert_to_tensor(padded_actions, dtype=tf.int64))

        return tf.convert_to_tensor(action_tensors, dtype=tf.int32), \
            tf.convert_to_tensor(action_mask_tensors, dtype=tf.float32), \
            tf.convert_to_tensor(action_idxs_tensors, dtype=tf.int32)

    @staticmethod
    def extract_elements_from_sequences(experience_sequences, key, config):
        return [utils.extract_from_batch(sequence_memory, key) for sequence_memory in experience_sequences]



