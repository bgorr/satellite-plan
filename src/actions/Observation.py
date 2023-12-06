import numpy as np
import os
import csv
import datetime
import multiprocessing
from functools import partial
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import time
import json
import pymap3d as pm
from datetime import datetime, timedelta
import math
from planners import utils




class Observation:


    @staticmethod
    def get_actions(sat, settings, top_n=5):
        curr_time = sat['sat_time']
        curr_angle = sat['sat_angle']
        obs_list = sat['all_obs']
        last_obs = sat['last_obs']

        # 1. Get next n possible observations
        feasible_actions = []
        for obs in obs_list:
            if last_obs is not None and obs["location"]["lat"] == last_obs["location"]["lat"]:
                continue
            if obs["start"] > curr_time:
                feasible, transition_end_time = Observation.check_maneuver_feasibility(
                    curr_angle,
                    np.min(obs["angles"]),
                    curr_time,
                    obs["end"],
                    settings
                )
                if transition_end_time < obs["start"]:
                    obs["soonest"] = obs["start"]
                else:
                    obs["soonest"] = transition_end_time

                if feasible:
                    obs_action = deepcopy(obs)
                    obs_action['mem-overflow'] = False
                    obs_action['event-seen'] = False
                    obs_action['novel-event-seen'] = False
                    feasible_actions.append(obs_action)
            if len(feasible_actions) >= top_n:
                break

        # 2. Determine next state (storage) / reward if observation is taken
        for obs in feasible_actions:
            obs['type'] = 'N'  # Either: N for none | E for event | NE for novel event
            obs['reward'] = deepcopy(settings['obs-reward'])

            # --> Determine if event is observed
            g_point = utils.get_gridpoint(obs['gp_index'], settings)
            for event in g_point['events']:
                if event['start'] <= obs['soonest'] <= event['end']:
                    if event['times_observed'] == 0:
                        obs['reward'] += settings['obs-novel-event-reward']
                        obs['type'] = 'NE'
                    else:
                        obs['reward'] += settings['obs-event-reward']
                        obs['type'] = 'E'

            # --> Determine new state
            obs['storage'] = deepcopy(sat['storage'])

            # --> Only if using storage state
            # if len(sat['storage']) < settings['sat-storage-cap']:
            #     obs['storage'].append(obs['type'])
            # else:
            #     obs['reward'] += settings['mem-overflow-penalty']
            #     obs['mem-overflow'] = True

        return feasible_actions


    @staticmethod
    def check_maneuver_feasibility(curr_angle, obs_angle, curr_time, obs_end_time, settings):
        """
        Checks to see if the specified angle change violates the maximum slew rate constraint.
        """
        if (obs_end_time == curr_time):
            return False, False
        slew_rate = abs(obs_angle - curr_angle) / abs(obs_end_time - curr_time) / settings["step_size"]
        max_slew_rate = settings["agility"]  # deg / s
        transition_end_time = abs(obs_angle - curr_angle) / (max_slew_rate * settings["step_size"]) + curr_time
        return slew_rate < max_slew_rate, transition_end_time


