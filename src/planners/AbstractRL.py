import numpy as np
import os
import csv
import json
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


# Utility functions
from planners import utils


class AbstractRL:

    # ---------------------------------------------
    # Initialize
    # ---------------------------------------------

    def __init__(self, settings):
        self.pool_size = 32  # config.cores
        self.directory = settings["directory"] + "orbit_data/"
        self.settings = settings

        # Hyperparameters
        self.num_epochs = 1
        self.target_update_frequency = 3
        self.replay_batch_size = 32
        self.replay_frequency = 1
        self.buffer_init_size = 50
        self.clip_gradients = False

        # 1. Initialize satellites / events / grid locations
        print('--> INITIALIZING SATS')
        self.satellites = self.init_sats()
        self.init_models()
        self.events = self.init_events()  # Must be init after satellites
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
        self.step_observations = [0]
        self.sat_observations_left = []
        self.sat_timesteps = []
        self.step_events = [0]
        self.steps = 0

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
        for subdir_idx, subdir in enumerate(os.listdir(self.directory)):
            satellite = {
                'sat_time': 0.0,      # Current timestep after last action (init to 0)
                'sat_angle': 0.0,     # Current slewing angle after last action
                'sat_lat': None,
                'sat_lon': None,
                'last_obs': None,     # The last observation taken by the satellite
                'has_actions': True,  # Records if sat has any actions left to take
                'location_list': [],  # Locations seen by sat

                'last_action_idx': None,  # The index of the last action taken (for delayed back-prop)
                'last_state': None,       # The input state for the last action taken (for delayed back-prop)
                'rewards': [],            # All rewards received by the satellite (for delayed back-prop)
                'took_action':  False,    # Records if the satellite took an action in the last iteration (for delayed back-prop)

                'all_obs': [],  # All observations taken by the satellite (for csv metrics)

                # Metrics for reward
                'last_event_time': 0.0,
                'obs_left': -1,
                'obs_left_store': [],
                'constrained_actions': [],
                'sat_id': len(all_sats),
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

                if "state_geo" in f:
                    with open(self.directory + subdir + "/" + f, newline='') as csv_file:
                        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                        nadir_lat_lons = []
                        for idx, row in enumerate(spamreader):
                            if idx < 1:
                                continue
                            row = [float(i) for i in row]
                            nadir_lat_lons.append([row[0], row[1], row[2]])
                        satellite["nadir_lat_lons"] = nadir_lat_lons

            satellite['sat_lat'] = satellite["nadir_lat_lons"][0][1]
            satellite['sat_lon'] = satellite["nadir_lat_lons"][0][2]
            all_sats.append(satellite)

        return all_sats

    def init_models(self):
        for satellite in self.satellites:
            satellite['q_network'] = SatelliteMLP().implicit_build()
            satellite['target_q_network'] = SatelliteMLP().implicit_build()

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
                        "severity": float(row[4]),
                        "times_observed": 0,
                        "satellite_discoveries": [-1] * len(self.satellites),
                        "rarity": 1,
                        "reward": 1,
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
            AbstractRL.write_sat_and_save(idx, "initial_obs_rl", len(sat_windows[idx]))
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

            # 1. Take a step in environment
            # self.sync_step(actionable_sats, counter)
            took_action = self.async_step(actionable_sats, counter)
            if took_action is False:
                break

            if counter % 10 == 0:
                sat_steps = [round(sat['q_network'].step / (counter+2), 2) for sat in self.satellites]
                sat_steps_avg = round(sum(sat_steps) / len(sat_steps), 2)
                print('UPDATES', counter, ' AVG | ACTIONS :',  sat_steps_avg)

            # 2. Update satellite networks
            min_memory_len = min([len(sat['q_network'].memory) for sat in self.satellites])
            if min_memory_len > self.buffer_init_size:
                self.update_satellite_models()

            # 3. Reset action tracker
            for sat in self.satellites:
                sat['took_action'] = False

            # 4. Regather actionable sats
            actionable_sats = [sat for sat in self.satellites if sat['has_actions'] is True]

            # 5. Copy over q_network to target_q_network
            if counter % self.target_update_frequency == 0:
                for sat in self.satellites:
                    sat['target_q_network'].load_target_weights(sat['q_network'])

            # 6. Plot results
            if counter > 0 and counter % self.plot_frequency == 0:
                self.plot_sat_timesteps()
                # self.plot_sat_actions_left()
                self.plot_results(counter)

            # 7. Break if debugging
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


            # if counter > 100:
            #     break

    def satellite_action(self, satellite, debug=False):
        pass

    def record_satellite_experience(self):
        pass

    def update_satellite_models(self):
        pass

    # ---------------------------------------------
    # Step methods
    # ---------------------------------------------

    def sync_step(self, actionable_sats, counter):
        for idx, sat in enumerate(actionable_sats):
            debug = False
            if idx == 0 and counter % 10 == 0:
                debug = True
            # init = counter < self.buffer_init_size
            self.satellite_action(sat, debug)  # 0.02 seconds per satellite
            self.step_observations[counter] += 1

    def async_step(self, actionable_sats, counter):
        action_taken = False
        if counter == 0:
            for idx, sat in enumerate(actionable_sats):
                self.satellite_action(sat, False)
                self.step_observations[counter] += 1
                action_taken = True
        else:
            all_sat_times = [sat['sat_time'] for sat in actionable_sats]
            max_sat_time = max(all_sat_times)
            max_sat_idx = all_sat_times.index(max_sat_time)
            for idx, sat in enumerate(actionable_sats):
                if idx == max_sat_idx:
                    continue
                sat_time = sat['sat_time']
                debug = (counter % 10 == 0) and (idx == 0)
                while sat_time < max_sat_time and sat['has_actions'] is True:
                    self.satellite_action(sat, debug)
                    self.step_observations[counter] += 1
                    sat_time = sat['sat_time']
                    action_taken = True
        return action_taken

    # ---------------------------------------------
    # Get action space
    # ---------------------------------------------

    def get_action_space(self, curr_time, curr_angle, obs_list, last_obs, settings):
        feasible_actions = []
        if last_obs:
            last_obs_lat = {last_obs["location"]["lat"]}  # Using a set for faster lookup
        agility = settings["agility"]
        step_size = settings["step_size"]
        obs_left = None
        for idx, obs in enumerate(obs_list):
            if last_obs and obs["location"]["lat"] in last_obs_lat:
                continue
            if obs["start"] > curr_time:
                if obs_left is None:
                    obs_left = len(obs_list) - idx

                feasible, transition_end_time = self.check_maneuver_feasibility(
                    curr_angle, np.min(obs["angles"]), curr_time, obs["end"], agility, step_size
                )
                if feasible:
                    obs["soonest"] = max(obs["start"], transition_end_time)
                    feasible_actions.append(obs)

                    # obs_copy = deepcopy(obs)
                    # obs_copy["soonest"] = max(obs_copy["start"], transition_end_time)
                    # feasible_actions.append(obs_copy)

                if len(feasible_actions) >= 10:  # Break out early if N feasible actions are found
                    break

        return feasible_actions, obs_left

    def get_constrained_actions(self, curr_time, curr_angle, obs_list, last_obs, settings):
        feasible_actions = []
        if last_obs:
            last_obs_lat = {last_obs["location"]["lat"]}  # Using a set for faster lookup
        agility = settings["agility"]
        step_size = settings["step_size"]
        constrained_actions = 0
        for idx, obs in enumerate(obs_list):
            if last_obs and obs["location"]["lat"] in last_obs_lat:
                continue
            if obs["start"] > curr_time:
                feasible, transition_end_time = self.check_maneuver_feasibility(
                    curr_angle, np.min(obs["angles"]), curr_time, obs["end"], agility, step_size
                )
                if feasible is False:
                    constrained_actions += 1

                if feasible:
                    obs["soonest"] = max(obs["start"], transition_end_time)
                    feasible_actions.append(obs)
                if len(feasible_actions) >= 5:  # Break out early if N feasible actions are found
                    break

        return constrained_actions

    def check_maneuver_feasibility(self, curr_angle, obs_angle, curr_time, obs_end_time, max_slew_rate, step_size):
        if obs_end_time == curr_time:
            return False, False

        # Determine if the maneuver is feasible
        slew_rate_steps = abs(obs_angle - curr_angle) / abs(obs_end_time - curr_time)  # deg / steps
        slew_rate_secs = slew_rate_steps / step_size  # deg / sec
        can_slew = True
        if slew_rate_secs > max_slew_rate:
            # print('--> SLEW CONSTRAINED POINT')
            can_slew = False

        # Determine when the maneuver will end
        max_slew_rate_steps = max_slew_rate * step_size  # deg / step
        transition_time = abs(obs_angle - curr_angle) / max_slew_rate_steps  # steps
        transition_end_time = transition_time + curr_time  # steps

        return can_slew, transition_end_time

    def find_event_bonus(self, best_obs, satellite):
        for event in self.events:
            if utils.close_enough(best_obs["location"]["lat"], best_obs["location"]["lon"], event["location"]["lat"], event["location"]["lon"]):
                if (event["start"] <= best_obs["start"] <= event["end"]) or (event["start"] <= best_obs["end"] <= event["end"]):
                    obs_time = best_obs['start']
                    base_reward = event['reward']

                    # Reward Func Terms
                    # 1. Temporal penalty: add penalty for time until first obs (max time val ~ 8.6k for one day)
                    # 2. Discovery bonus: add bonus for newly discovered events
                    # 3. Rarity bonus: add bonus for rare events
                    # 4. Collaboration bonus: add bonus for multiple satellites observing the same event

                    temporal_penalty = (obs_time / 10000.0) * -1.0

                    sat_discovery_time = event['satellite_discoveries'][satellite['sat_id']]
                    if sat_discovery_time == -1:
                        event['satellite_discoveries'][satellite['sat_id']] = obs_time
                        event['times_observed'] += 1.0
                        discovery_bonus = 1.0
                    else:
                        discovery_bonus = 0.0

                    rarity_bonus = 1.0 / event['rarity']
                    collaboration_bonus = 1 - (1.0 / event['times_observed'])

                    reward = base_reward
                    return reward
        return 0.0

    # ---------------------------------------------
    # Record results
    # ---------------------------------------------

    def record_planning_results(self, epoch):
        for idx, sat in enumerate(self.satellites):
            to_write = len(sat['q_network'].memory)
            AbstractRL.write_sat_and_save(idx, "action_count_rl", to_write)
            timestamps_rl = [mem[0] for mem in sat['q_network'].memory]
            AbstractRL.write_sat_and_save(idx, "timestamps_rl", timestamps_rl)
            AbstractRL.write_sat_and_save(idx, "obs_left_rl", sat['obs_left_store'])


        # Gather satellite args
        satellites_with_args = [(sat["orbitpy_id"], sat['all_obs'], self.settings, self.grid_locations, epoch) for sat in self.satellites]

        # Create a pool of worker processes
        num_processes = self.pool_size  # Get the number of CPU cores
        with Pool(num_processes) as pool:
            list(tqdm(pool.imap(AbstractRL.record_satellite, satellites_with_args), total=len(self.satellites),
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
        cumulative_obs = [sum(self.step_observations[:i + 1]) for i in range(len(self.step_observations))]
        cumulative_rewards = [sum(self.step_events[:i + 1]) for i in range(len(self.step_events))]
        plt.plot(cumulative_obs, cumulative_rewards)
        plt.xlabel('Total Observations')
        plt.ylabel('Total Events')
        plt.title('Step Events vs Step Observations')
        plt.show()


    def plot_sat_actions_left(self):
        transposed_obs_left = list(map(list, zip(*self.sat_observations_left)))
        timesteps = list(range(1, len(self.sat_observations_left) + 1))
        for idx, satellite_data in enumerate(transposed_obs_left):
            plt.plot(timesteps, satellite_data, label=f'Satellite {idx + 1}')
        plt.xlabel('Async Steps')
        plt.ylabel('Observations Left')
        plt.title('Number of Observations Left vs Timesteps')
        plt.show()


    def plot_sat_timesteps(self):
        transposed_timesteps = list(map(list, zip(*self.sat_timesteps)))
        timesteps = list(range(1, len(self.sat_timesteps) + 1))
        if len(timesteps) > 100:
            timesteps = timesteps[-100:]
        for idx, satellite_data in enumerate(transposed_timesteps):
            if len(satellite_data) > 100:
                satellite_data = satellite_data[-100:]
            plt.plot(timesteps, satellite_data, label=f'Satellite {idx + 1}')
        plt.xlabel('Async Steps')
        plt.ylabel('Sat Time')
        plt.title('Sat mission time vs RL Timesteps')
        plt.show()


    @staticmethod
    def write_sat_and_save(sat_num, key, value):
        sat_file = os.path.join(config.results_dir, f"sat_{sat_num}.json")
        with open(sat_file, "r") as f:
            sat_json = json.load(f)
        sat_json[key] = value
        with open(sat_file, 'w') as f:
            json.dump(sat_json, f, indent=4)









