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


class BaseRL(AbstractRL):

    def __init__(self, settings):
        super().__init__(settings)

        self.pool_size = 36  # config.cores
        self.directory = settings["directory"] + "orbit_data/"
        self.settings = settings

        # Hyperparameters
        self.num_epochs = 1
        self.target_update_frequency = 5
        self.replay_batch_size = 128
        self.replay_frequency = 1
        self.buffer_init_size = 100
        self.clip_gradients = False

        # 3. Optimizers
        self.gamma = 0.9

        # 4. Metrics
        self.plot_frequency = 100
        self.sample_time_deltas = []
        self.sample_async_terms = []


    def init_models(self):
        for satellite in self.satellites:
            satellite['q_network'] = SatelliteMLP().implicit_build()
            satellite['target_q_network'] = SatelliteMLP().implicit_build()

    # --------------------------------------------
    # Train Planners
    # --------------------------------------------

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
        satellite['obs_left_store'].append(deepcopy(obs_left))

        # 2. Create q_network state, record
        state = deepcopy([sat_time, sat_angle])
        satellite['last_state'] = state
        # print('--> STATE', state)

        # 3. Get q_network action, record
        q_network = satellite['q_network']
        action_idx = q_network.get_action(state, num_actions=num_actions, debug=debug, rand_action=False)
        satellite['last_action_idx'] = action_idx
        # print('--> ACTION SELECTED:', action_idx)

        # 4. Record selected action
        action = actions[action_idx]
        satellite['all_obs'].append(deepcopy(action))
        satellite['location_list'].append(action["location"])  # already seen by this sat
        satellite['last_obs'] = deepcopy(action)
        satellite['sat_time'] = action["soonest"]
        satellite['sat_angle'] = action["angle"]
        satellite['took_action'] = True
        satellite['constrained_actions'] = self.get_constrained_actions(
            satellite['sat_time'],
            satellite['sat_angle'],
            satellite['obs_list'],
            satellite['last_obs'],
            self.settings
        )
        # print('--> CONSTRAINED ACTIONS:', satellite['constrained_actions'])

        # 5. Record new lat and lon
        # sat_time_step_approx = int(round(action["soonest"], 0))
        # nadir_lat_lons = satellite['nadir_lat_lons'][sat_time_step_approx]
        # time, lat, lon = nadir_lat_lons[0], nadir_lat_lons[1], nadir_lat_lons[2]
        # satellite['sat_lat'] = lat
        # satellite['sat_lon'] = lon

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
            [satellite['sat_time'], satellite['sat_angle']]
        )

        # 9. Record reward
        satellite['rewards'].append(deepcopy(reward))
        return True

    def update_satellite_models(self):

        # 0. Get satellites that took actions
        prep_start = time.time()

        # trainable_sats = [sat for sat in self.satellites if sat['took_action'] is True]
        trainable_sats = self.satellites

        # 1. Sample experiences from experience replay buffer
        # batch of n experiences for each satellite
        # experience: state, action, reward, next_state
        # sat_experiences shape: (num_sats, batch_size, experience_dim)
        async_terms = None
        sat_ref_times = None  # (num_sats, batch_size,)
        # sat_experiences = self.sample_experiences(trainable_sats)
        sat_experiences, async_terms, ref_times = self.sample_async_experiences(trainable_sats)
        sat_ref_times = [ref_times for _ in range(len(trainable_sats))]

        # 2. Get cumulative reward
        # all_rewards = []  # (num_sats, batch_size)
        # for experience in sat_experiences:
        #     batch_rewards = utils.extract_from_batch(experience, 'reward')
        #     all_rewards.append(batch_rewards)
        # cumulative_reward = tf.reduce_sum(all_rewards, axis=0)  # shape: (batch_size)


        batch_states = [  # (num_sats, batch_size, state_dim)
            tf.convert_to_tensor(utils.extract_from_batch(experience, 'state'), dtype=tf.float32) for experience in sat_experiences
        ]
        batch_actions = [  # (num_sats, batch_size, action_dim)
            tf.convert_to_tensor(utils.extract_from_batch(experience, 'action'), dtype=tf.int32) for experience in sat_experiences
        ]
        batch_next_states = [  # (num_sats, batch_size, state_dim)
           tf.convert_to_tensor(utils.extract_from_batch(experience, 'next_state'), dtype=tf.float32) for experience in sat_experiences
        ]
        batch_rewards = [  # (num_sats, batch_size)
            tf.convert_to_tensor(utils.extract_from_batch(experience, 'reward'), dtype=tf.float32) for experience in sat_experiences
        ]

        # Apply async terms
        if async_terms:
            batch_async_rewards = []
            for idx, satellite_batch_rewards in enumerate(batch_rewards):
                async_term_tensor = tf.convert_to_tensor(async_terms[idx], dtype=tf.float32)
                # print(async_term_tensor)
                sat_async_rewards = satellite_batch_rewards * async_term_tensor
                batch_async_rewards.append(sat_async_rewards)
            batch_rewards = batch_async_rewards

        # Cumulative reward across sat batches
        cumulative_reward = tf.reduce_sum(batch_rewards, axis=0)  # shape: (batch_size,)


        # Find target q_values for next state
        target_q_values = []  # shape: (num_sats, batch_size)
        for idx, sat in enumerate(trainable_sats):
            # INPUT: next_states (batch_size, 2)
            # OUTPUT: target_q_values (batch_size,)
            target_q_value = sat['target_q_network'].get_q_value_batch_max(batch_next_states[idx])
            target_q_values.append(target_q_value)
        target_q_tensor = tf.stack(target_q_values)  # shape: (num_sats, batch_size)
        summed_q_value_next = tf.reduce_sum(target_q_tensor, axis=0)  # shape: (batch_size,)
        target_q_value = cumulative_reward + self.gamma * summed_q_value_next  # shape: (batch_size,)
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
        dataset = tf.data.Dataset.from_tensor_slices((batch_states, batch_actions, target_q_value_inputs, external_sat_sum))

        # Train
        losses = []
        for idx, data_tuple in enumerate(dataset):
            batch_states, batch_actions, target_q_value, current_q_values = data_tuple
            metrics = trainable_sats[idx]['q_network'].train_step(batch_states, batch_actions, target_q_value, current_q_values)
            losses.append(metrics['loss'])

        print('--> AVERAGE LOSS:', np.mean(losses))
        # print('SAMPLE TIME DELTA MEAN/VAR:', round(np.mean(self.sample_time_deltas), 3), round(np.var(self.sample_time_deltas), 3))
        # rand_idx = random.randint(0, len(self.sample_time_deltas)-1)
        # print('Sample time delta:', self.sample_time_deltas[rand_idx], 'async term:', self.sample_async_terms[rand_idx])

    def sample_experiences(self, trainable_sats):
        min_buf_size = 1e9
        for sat in trainable_sats:
            if len(sat['q_network'].replay_buffer) < min_buf_size:
                min_buf_size = len(sat['q_network'].replay_buffer)

        sat_experiences = []  # (num_sats, batch_size, 5)
        indices = random.sample(range(int(min_buf_size)), self.replay_batch_size)
        for sat in trainable_sats:
            sat_experiences.append(sat['q_network'].sample_buffer(self.replay_batch_size, indices=indices))
        return sat_experiences

    def sample_async_experiences(self, trainable_sats):

        def closest_time(target_time, time_list):
            closest_time = min(time_list, key=lambda x: abs(x - target_time))
            closest_time_idx = time_list.index(closest_time)
            return closest_time, closest_time_idx

        def closest_future_time(target_time, time_list):
            # Filter out the times that are in the past relative to the target_time
            future_times = [time for time in time_list if time >= target_time]

            # If there are no future times, return None
            if not future_times:
                closest_time = min(time_list, key=lambda x: abs(x - target_time))
                closest_time_idx = time_list.index(closest_time)
                return closest_time, closest_time_idx

            # Find the closest future time
            closest_time = min(future_times, key=lambda x: abs(x - target_time))
            closest_time_idx = time_list.index(closest_time)
            return closest_time, closest_time_idx

        ragged_memory_matrix = [sat['q_network'].get_memory_time_values() for sat in self.satellites]


        samples = []
        samples_async_terms = []
        samples_ref_times = []  # (batch_size,)
        for _ in range(self.replay_batch_size):
            # select a random reference sat OR use same sat to reduce variance
            ref_sat_idx = random.randint(0, len(self.satellites)-1)
            # ref_sat_idx = 0
            ref_sat_memory_idx = random.randint(0, len(ragged_memory_matrix[ref_sat_idx])-1)
            ref_sat_memory_time = ragged_memory_matrix[ref_sat_idx][ref_sat_memory_idx]

            sample = []
            sample_times = []
            sample_ref_time = None
            async_terms = []
            for idx, sat in enumerate(self.satellites):
                if idx == ref_sat_idx:
                    sample.append(ref_sat_memory_idx)
                    sample_times.append(ref_sat_memory_time)
                    async_terms.append(1.0)
                    sample_ref_time = ref_sat_memory_time
                else:
                    # c_time, c_time_idx = closest_time(ref_sat_memory_time, ragged_memory_matrix[idx])
                    c_time, c_time_idx = closest_future_time(ref_sat_memory_time, ragged_memory_matrix[idx])

                    ref_time_diff = ref_sat_memory_time - c_time
                    ref_time_delta = abs(ref_sat_memory_time - c_time)
                    self.sample_time_deltas.append(ref_time_diff)
                    async_term = np.exp((np.log(0.1) * ref_time_delta) / 20)
                    async_terms.append(async_term)
                    self.sample_async_terms.append(async_term)
                    sample.append(c_time_idx)
                    sample_times.append(c_time)
            samples.append(sample)
            samples_async_terms.append(async_terms)
            samples_ref_times.append(sample_ref_time)

        # Transpose samples such that satellite samples are grouped
        satellite_samples = []
        for samp in samples:
            for idx, sat in enumerate(self.satellites):
                if len(satellite_samples) < idx + 1:
                    satellite_samples.append([])
                satellite_samples[idx].append(samp[idx])

        satellite_samples_async_terms = []
        for samp in samples_async_terms:
            for idx, sat in enumerate(self.satellites):
                if len(satellite_samples_async_terms) < idx + 1:
                    satellite_samples_async_terms.append([])
                satellite_samples_async_terms[idx].append(samp[idx])


        # Get experiences
        sat_experiences = []
        for idx, indices in enumerate(satellite_samples):
            sat_experiences.append(self.satellites[idx]['q_network'].sample_buffer(self.replay_batch_size, indices=indices))
        return sat_experiences, satellite_samples_async_terms, samples_ref_times











