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
            satellite['q_network'] = SatelliteMLP().implicit_build()
            satellite['target_q_network'] = SatelliteMLP().implicit_build()

    # --------------------------------------------
    # Train Planners
    # --------------------------------------------

    def satellite_action(self, satellite, debug=False, init=False):
        obs_list = satellite['obs_list']
        sat_time = satellite['sat_time']
        sat_angle = satellite['sat_angle']
        last_obs = satellite['last_obs']
        sat_lat = satellite['sat_lat']
        sat_lon = satellite['sat_lon']

        # 1. Get action space (size 10)
        actions = self.get_action_space(sat_time, sat_angle, obs_list, last_obs, self.settings)
        num_actions = len(actions)
        if num_actions == 0:
            satellite['has_actions'] = False
            satellite['took_action'] = False
            return

        # 2. Create q_network state, record
        state = deepcopy([sat_time, sat_angle])
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
        last_action_indices = []
        for sat in trainable_sats:
            last_action_idx = sat['last_action_idx']
            last_action_indices.append(last_action_idx)

        # 3. (target_q_network) Find summed q_values for "next" state
        target_states = []
        for sat in trainable_sats:
            curr_state = [sat['sat_time'], sat['sat_angle']]
            target_states.append(curr_state)

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

        trainable_sats = [sat for sat in self.satellites if sat['took_action'] is True]

        # 1. Sample experiences from experience replay buffer
        min_buf_size = 1e9
        for sat in trainable_sats:
            if len(sat['q_network'].replay_buffer) < min_buf_size:
                min_buf_size = len(sat['q_network'].replay_buffer)

        # print('--> MIN BUF SIZE:', min_buf_size)
        sat_experiences = []  # (num_sats, batch_size, 5)
        indices = random.sample(range(int(min_buf_size)), self.replay_batch_size)
        for sat in trainable_sats:
            sat_experiences.append(sat['q_network'].sample_buffer(self.replay_batch_size, indices=indices))

        # 2. Get cumulative reward
        all_rewards = []  # (num_sats, batch_size, 1)
        for experience in sat_experiences:
            batch_rewards = utils.extract_from_batch(experience, 'reward')
            all_rewards.append(batch_rewards)
        cumulative_reward = tf.reduce_sum(all_rewards, axis=0)  # shape: (batch_size)

        batch_states = [
            tf.convert_to_tensor(utils.extract_from_batch(experience, 'state'), dtype=tf.float32) for experience in sat_experiences
        ]
        batch_actions = [
            tf.convert_to_tensor(utils.extract_from_batch(experience, 'action'), dtype=tf.int32) for experience in sat_experiences
        ]
        batch_next_states = [
           tf.convert_to_tensor(utils.extract_from_batch(experience, 'next_state'), dtype=tf.float32) for experience in sat_experiences
        ]

        # Find target q_values for next state
        target_q_values = []  # shape: (num_sats, batch_size)
        for idx, sat in enumerate(trainable_sats):
            # for batch_state, batch_action, batch_next_state in dataset:
            target_q_value = sat['target_q_network'].get_q_value_batch_max(batch_next_states[idx])  # shape: (batch_size,)
            target_q_values.append(target_q_value)
        target_q_tensor = tf.stack(target_q_values)  # shape: (num_sats, batch_size)
        summed_q_value_next = tf.reduce_sum(target_q_tensor, axis=0)  # shape: (batch_size,)
        target_q_value = cumulative_reward + self.gamma * summed_q_value_next  # shape: (batch_size,)
        target_q_value = tf.cast(target_q_value, dtype=tf.float32)
        target_q_value_inputs = [target_q_value] * len(trainable_sats)

        # Find summed q_values for last (actually current) state
        last_q_values = []  # shape: (num_sats, batch_size)
        for idx, sat in enumerate(trainable_sats):
            batch_experience = sat_experiences[idx]
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
        curr_time = time.time()
        for idx, data_tuple in enumerate(dataset):
            batch_states, batch_actions, target_q_value, current_q_values = data_tuple
            trainable_sats[idx]['q_network'].train_step(batch_states, batch_actions, target_q_value, current_q_values)
        # print('--> PREP / TRAINING TIME:', time.time() - prep_start, time.time() - curr_time)












