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
import tensorflow as tf
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt
import config
from deepdiff import DeepDiff

# Utility functions
from planners import utils


class BaseRL:

    def __init__(self, settings):
        self.pool_size = config.cores
        self.directory = settings["directory"] + "orbit_data/"
        self.settings = settings

        # Hyperparameters
        self.num_epochs = 1
        self.target_update_frequency = 10
        self.replay_batch_size = 64
        self.replay_frequency = 3
        self.buffer_init_size = 100
        self.clip_gradients = False

        # 1. Initialize satellites / events / grid locations
        print('--> INITIALIZING SATS')
        self.satellites = self.init_sats()
        self.events = self.init_events()
        self.grid_locations = self.init_grid_locations()

        # 2. Initialize observations
        print('--> INITIALIZING OBSERVATIONS')
        self.init_observations()

        # 3. Optimizers
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.gamma = 0.9

        # 4. Metrics
        self.plot_frequency = 250
        self.step_rewards = []
        self.step_observations = []

    # ---------------------------------------------
    # Initialize
    # ---------------------------------------------

    def init_grid_locations(self):
        grid_locations = []
        with open(self.settings["point_grid"], 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvfile)
            for row in csvreader:
                grid_locations.append([float(row[0]), float(row[1])])
        return grid_locations

    def init_sats(self):
        all_sats = []
        for subdir in os.listdir(self.directory):
            satellite = {
                'sat_time': 0.0,      # Current time after last action (init to 0)
                'sat_angle': 0.0,     # Current slewing angle after last action
                'last_obs': None,     # The last observation taken by the satellite
                'has_actions': True,  # Records if sat has any actions left to take
                'location_list': [],  # Locations seen by sat

                'last_action_idx': None,  # The index of the last action taken (for delayed back-prop)
                'last_state': None,       # The input state for the last action taken (for delayed back-prop)
                'rewards': [],            # All rewards received by the satellite (for delayed back-prop)
                'took_action':  False,    # Records if the satellite took an action in the last iteration (for delayed back-prop)

                'all_obs': [],  # All observations taken by the satellite (for csv metrics)
            }
            if "comm" in subdir:
                continue
            if ".json" in subdir:
                continue
            satellite['files'] = os.listdir(self.directory + subdir)
            for f in satellite['files']:
                if "datametrics" in f:
                    with open(self.directory + subdir + "/" + f, newline='') as csv_file:
                        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                        visibilities = []
                        i = 0
                        for row in spamreader:
                            if i < 5:
                                i = i + 1
                                continue
                            row[2] = "0.0"
                            row = [float(i) for i in row]
                            visibilities.append(row)
                    satellite["visibilities"] = visibilities
                    satellite["orbitpy_id"] = subdir
                    satellite['q_network'] = SatelliteMLP().implicit_build()
                    satellite['target_q_network'] = SatelliteMLP().implicit_build()
            all_sats.append(satellite)
        return all_sats

    def init_events(self):
        events = []
        for event_filename in self.settings["event_csvs"]:
            with open(event_filename, newline='') as csv_file:
                csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                i = 0
                for row in csvreader:
                    if i < 1:
                        i = i + 1
                        continue
                    event = {
                        "location": {
                            "lat": float(row[0]),
                            "lon": float(row[1]),
                        },
                        "start": float(row[2]) / self.settings["step_size"],
                        "end": (float(row[2]) + float(row[3])) / self.settings["step_size"],
                        "severity": float(row[4])
                    }
                    events.append(event)
        print('--> TOTAL EVENTS:', len(events))
        return events

    def init_observations(self):
        ''' Visibility Structure
            visible_point = [
                1.0,     # time index
                3322.0,  # GP index
                0.0,     # pnt-opt index
                3.496,   # lat [deg]
                -99.38,  # lon [deg]
                605.7,   # observation range [km]
                -33.04,  # look angle [deg]
                36.01,   # incidence angle [deg]
                82.11    # solar zenith [deg]
            ]
        '''
        all_visibilities = [sat['visibilities'] for sat in self.satellites]
        with Pool(processes=self.pool_size) as pool:
            sat_windows = list(tqdm(pool.imap(utils.init_sat_observations, all_visibilities), total=len(all_visibilities)))
        for idx, sat in enumerate(self.satellites):
            sat['obs_list'] = sat_windows[idx]

    # --------------------------------------------
    # Train Planners
    # --------------------------------------------
    #
    # Problem Definition
    # - mission duration: 1 day (86400 seconds)
    # - ground points: N distributed points on the globe
    # - events: unknown subset of ground points E satellites get rewarded for observing
    # - sat1: has potential set of ground point observations P1 (time-ordered)
    # --- slewing constraints only allow it to observe a subset of P1, given by S1
    # --- the previous observation determines the possible set of next observations
    # Satellite 1 Timestep Simulation
    # - INIT: satellite can decide to observe any ground point in P1 = [1, 2, ..., M]
    # 1. (sat chooses to observe point 2) --> (new current_time for sat1 is time at which it observes point 2)
    # 2. (calculate new possible set of observations P1 considering slewing constraint) --> P1 = [4, 5, ..., M]

    def train_planners(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs

        for epoch in range(num_epochs):

            # 1. Plan mission
            self.plan_mission()

            # 2. Record planning results
            self.record_planning_results(epoch)

    def plan_mission(self):

        # 1. Reset all satellite observations
        for sat in self.satellites:
            sat['all_obs'] = []

        # 2. Gather sats that have actions left to take
        counter = 0
        actionable_sats = [sat for sat in self.satellites if sat['has_actions'] is True]
        while(len(actionable_sats) > 0):

            # 3. Take an action for each actionable sat
            for idx, sat in enumerate(actionable_sats):
                debug = False
                if idx == 0 and counter % 10 == 0:
                    debug = True
                self.satellite_action(sat, debug)  # 0.2 seconds per satellite

            # 4. Update satellite policies
            # self.update_satellite_policies()  # 0.8 seconds (not bad)
            self.record_satellite_experience()
            if counter > self.replay_batch_size and counter % self.replay_frequency == 0 and counter > self.buffer_init_size:
                self.update_satellite_policies_experience_replay()

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
            # if counter > 50:
            #     break

    def satellite_action(self, satellite, debug=False):
        obs_list = satellite['obs_list']
        sat_time = satellite['sat_time']
        sat_angle = satellite['sat_angle']
        last_obs = satellite['last_obs']

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
        action_idx = q_network.get_aciton(state, num_actions=num_actions, debug=debug)
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

        # 5. Calculate action reward
        reward = action["reward"]
        reward += self.find_event_bonus(action, satellite['sat_time'])
        satellite['rewards'].append(deepcopy(reward))

    def find_event_bonus(self, best_obs, curr_time):
        for event in self.events:
            if utils.close_enough(best_obs["location"]["lat"], best_obs["location"]["lon"], event["location"]["lat"], event["location"]["lon"]):
                if (event["start"] <= best_obs["start"] <= event["end"]) or (event["start"] <= best_obs["end"] <= event["end"]):
                    return event["severity"]
        return 0.0

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
        last_q_values = []
        last_action_indices = []
        for sat in trainable_sats:
            last_state = sat['last_state']
            last_action_idx = sat['last_action_idx']
            last_action_indices.append(last_action_idx)
            last_q_value = sat['q_network'].get_q_value(last_state, action_idx=last_action_idx)
            last_q_values.append(last_q_value)

        # 3. (target_q_network) Find summed q_values for "next" state
        target_q_values = []
        target_states = []
        for sat in trainable_sats:
            curr_state = [sat['sat_time'], sat['sat_angle']]
            target_states.append(curr_state)
            curr_q_value = sat['target_q_network'].get_q_value(curr_state)
            target_q_values.append(curr_q_value)

        # 4. Add to experience replay buffer
        for idx, sat in enumerate(trainable_sats):
            sat['q_network'].record_experience(
                sat['last_state'],
                last_action_indices[idx],
                all_rewards[idx],
                target_states[idx]
            )

    def update_satellite_policies(self):

        # 0. Get satellites that took actions
        trainable_sats = [sat for sat in self.satellites if sat['took_action'] is True]

        # 1. Get cumulative reward
        all_rewards = []
        for sat in trainable_sats:
            if sat['took_action'] is True and len(sat['rewards']) > 0:
                all_rewards.append(sat['rewards'][-1])
        cumulative_reward = sum(all_rewards)
        # print('--> CUMULATIVE REWARD:', cumulative_reward)

        # 2. Compute gradient
        with tf.GradientTape(persistent=True) as tape:

            # 1. Watch all trainable variables
            for sat in trainable_sats:
                tape.watch(sat['q_network'].trainable_variables)

            # 2. (q_network) Find summed q_values for last (actually current) state
            last_q_values = []
            last_action_indices = []
            for sat in trainable_sats:
                last_state = sat['last_state']
                last_action_idx = sat['last_action_idx']
                last_action_indices.append(last_action_idx)
                last_q_value = sat['q_network'].get_q_value(last_state, action_idx=last_action_idx)
                last_q_values.append(last_q_value)
            summed_q_value = tf.reduce_sum(last_q_values)

            # 3. (target_q_network) Find summed q_values for "next" state
            target_q_values = []
            target_states = []
            for sat in trainable_sats:
                curr_state = [sat['sat_time'], sat['sat_angle']]
                target_states.append(curr_state)
                curr_q_value = sat['target_q_network'].get_q_value(curr_state)
                target_q_values.append(curr_q_value)
            summed_q_value_next = tf.reduce_sum(target_q_values)

            # 4. Add to experience replay buffer
            for idx, sat in enumerate(trainable_sats):
                sat['q_network'].record_experience(sat['last_state'], last_action_indices[idx], all_rewards[idx], target_states[idx])

            # 5. Compute loss
            target_q_value = cumulative_reward + self.gamma * summed_q_value_next
            loss = tf.reduce_mean(tf.square(summed_q_value - target_q_value))
            print('--> Q-DIFFERENCE:', np.abs(target_q_value.numpy() - summed_q_value.numpy()), 'LOSS:', loss.numpy(), 'CUMULATIVE REWARD:', cumulative_reward)


        # 6. Compute gradients over all sats and apply
        # for idx, sat in enumerate(trainable_sats):
        #     gradients = tape.gradient(loss, sat['q_network'].trainable_variables)
        #
        #     # Clip the gradients by value
        #     if self.clip_gradients is True:
        #         gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        #
        #     sat['q_network'].optimizer.apply_gradients(zip(gradients, sat['q_network'].trainable_variables))

        all_trainable_variables = [var for sat in trainable_sats for var in sat['q_network'].trainable_variables]
        gradients = tape.gradient(loss, all_trainable_variables)
        if self.clip_gradients is True:
            gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, all_trainable_variables))

        del tape
        return loss

    def update_satellite_policies_experience_replay(self):

        # 0. Get satellites that took actions
        trainable_sats = [sat for sat in self.satellites if sat['took_action'] is True]

        # 1. Sample experiences from experience replay buffer
        sat_experiences = []
        min_buf_size = 1e9
        for sat in trainable_sats:
            if len(sat['q_network'].replay_buffer) < min_buf_size:
                min_buf_size = len(sat['q_network'].replay_buffer)

        # print('--> MIN BUF SIZE:', min_buf_size)
        indices = random.sample(range(min_buf_size), 32)
        for sat in trainable_sats:
            sat_experiences.append(sat['q_network'].sample_buffer(self.replay_batch_size, indices=indices))

        # 2. Get cumulative reward
        all_rewards = []
        for experience in sat_experiences:
            batch_rewards = utils.extract_from_batch(experience, 'reward')
            all_rewards.append(batch_rewards)
        cumulative_reward = tf.reduce_sum(all_rewards, axis=0)

        # 3. Compute gradient
        with tf.GradientTape(persistent=True) as tape:

            # 3.1. Watch all trainable variables
            for sat in trainable_sats:
                tape.watch(sat['q_network'].trainable_variables)

            # 3.2. (q_network) Find summed q_values for last (actually current) state
            last_q_values = []  # shape: (num_sats, batch_size)
            for idx, sat in enumerate(trainable_sats):
                batch_experience = sat_experiences[idx]
                batch_states = utils.extract_from_batch(batch_experience, 'state')
                batch_actions = utils.extract_from_batch(batch_experience, 'action')
                last_q_value = sat['q_network'].get_q_value_batch(batch_states, action_idxs=batch_actions)
                last_q_values.append(last_q_value)
            summed_q_value = tf.reduce_sum(last_q_values, axis=0)  # shape: (batch_size,)
            # print('last_q_values:', tf.convert_to_tensor(last_q_values))

            # 3.3. (target_q_network) Find summed q_values for "next" state
            target_q_values = [] # shape: (num_sats, batch_size)
            for idx, sat in enumerate(trainable_sats):
                batch_experience = sat_experiences[idx]
                batch_states = utils.extract_from_batch(batch_experience, 'next_state')
                targer_q_value = sat['target_q_network'].get_q_value_batch(batch_states)
                target_q_values.append(targer_q_value)
            summed_q_value_next = tf.reduce_sum(target_q_values, axis=0)  # shape: (batch_size,)
            # print('target_q_values:', tf.convert_to_tensor(target_q_values))

            # 3.4. Compute loss
            target_q_value = cumulative_reward + self.gamma * summed_q_value_next
            loss = tf.reduce_mean(tf.square(summed_q_value - target_q_value))
            # print('--> REPLAY LOSS:', loss.numpy())

        # 4. Compute gradients over all sats and apply
        # for idx, sat in enumerate(trainable_sats):
        #     gradients = tape.gradient(loss, sat['q_network'].trainable_variables)
        #
        #     # Clip the gradients by value
        #     if self.clip_gradients is True:
        #         gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        #
        #     sat['q_network'].optimizer.apply_gradients(zip(gradients, sat['q_network'].trainable_variables))


        all_trainable_variables = [var for sat in trainable_sats for var in sat['q_network'].trainable_variables]
        gradients = tape.gradient(loss, all_trainable_variables)
        if self.clip_gradients is True:
            gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, all_trainable_variables))
        del tape
        return loss

    # ---------------------------------------------
    # Get action space
    # ---------------------------------------------

    def get_action_space(self, curr_time, curr_angle, obs_list, last_obs, settings):
        feasible_actions = []
        if last_obs:
            last_obs_lat = {last_obs["location"]["lat"]}  # Using a set for faster lookup
        agility = settings["agility"]
        step_size = settings["step_size"]
        for obs in obs_list:
            if last_obs and obs["location"]["lat"] in last_obs_lat:
                continue
            if obs["start"] > curr_time:
                feasible, transition_end_time = self.check_maneuver_feasibility(
                    curr_angle, np.min(obs["angles"]), curr_time, obs["end"], agility, step_size
                )
                obs["soonest"] = min(obs["start"], transition_end_time)
                if feasible:
                    feasible_actions.append(obs)
                if len(feasible_actions) >= 10:  # Break out early if 10 feasible actions are found
                    break

        return feasible_actions

    def check_maneuver_feasibility(self, curr_angle, obs_angle, curr_time, obs_end_time, max_slew_rate, step_size):
        if obs_end_time == curr_time:
            return False, False
        slew_rate = abs(obs_angle - curr_angle) / ((obs_end_time - curr_time) * step_size)
        transition_end_time = abs(obs_angle - curr_angle) / (max_slew_rate * step_size) + curr_time
        return slew_rate < max_slew_rate, transition_end_time

    # ---------------------------------------------
    # Record results
    # ---------------------------------------------

    def record_planning_results(self, epoch):

        # Gather satellite args
        satellites_with_args = [(sat["orbitpy_id"], sat['all_obs'], self.settings, self.grid_locations, epoch) for sat in self.satellites]

        # Create a pool of worker processes
        num_processes = self.pool_size  # Get the number of CPU cores
        with Pool(num_processes) as pool:
            list(tqdm(pool.imap(BaseRL.record_satellite, satellites_with_args), total=len(self.satellites),
                      desc='Recording planning results'))

    @staticmethod
    def record_satellite(args):
        orbit_id, all_obs, settings, grid_locations, epoch = args

        sat_file = settings["directory"] + "orbit_data/" + orbit_id + f"/plan_rl_{str(epoch)}.csv"
        if os.path.exists(sat_file):
            os.remove(sat_file)

        # Precompute minimum FOV value
        min_fov = np.min([settings["cross_track_ffov"], settings["along_track_ffov"]])

        # Compute results
        results = []
        for obs in all_obs:
            in_fov = np.array([utils.within_fov_fast(loc, obs["location"], min_fov, 500) for loc in grid_locations])
            valid_locs = np.array(grid_locations)[in_fov]
            for loc in valid_locs:
                row = [obs["start"], obs["end"], loc[0], loc[1]]
                results.append(row)

        # Write to CSV file after all computations are done
        with open(sat_file, 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerows(results)

    # ---------------------------------------------
    # Plot results
    # ---------------------------------------------

    def plot_results(self, steps):
        cumulative_rewards = [sum(self.step_rewards[:i + 1]) for i in range(len(self.step_rewards))]
        # cumulative_observatoins = [sum(self.step_observations[:i + 1]) for i in range(len(self.step_observations))]
        # print('--> PLOTTING RESULTS', cumulative_rewards[-1])
        plt.plot(cumulative_rewards)
        plt.xlabel('Training Steps')
        plt.ylabel('Event Observations')
        plt.title('Event Observations (reward) over Training Steps')
        plt.show()














