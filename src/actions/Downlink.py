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



class Downlink:

    @staticmethod
    def get_actions(sat, settings, top_n=5):
        # 1. There should always exist a down-link action (except when no GS left in mission)
        # Case 1: GS in line of sight
        # Case 2: GS not in line of sight
        dl_action = deepcopy({
            "angle": sat['sat_angle'],
            "type": "down-link",
            "reward": 0,  # calculate reward
            "location": {
                "lat": 0.0,
                "lon": 0.0
            },
            "storage": deepcopy(sat['storage']),
        })

        # 2. Get ground station visits sorted by viz window start time
        gs_obs = sat['all_gs']
        gs_obs.sort(key=lambda x: x[0])
        gs_in_view = [(gs[0] <= sat['sat_time'] < gs[1]) for gs in gs_obs]
        if True in gs_in_view:
            next_gs = deepcopy(gs_obs[gs_in_view.index(True)])
            dl_time = next_gs[1] - sat['sat_time']
            dl_start = sat['sat_time']
        else:
            future_gs = [gs for gs in gs_obs if sat['sat_time'] < gs[0]]
            if len(future_gs) == 0:
                return []
            next_gs = deepcopy(future_gs[0])
            dl_time = next_gs[1] - next_gs[0]
            dl_start = next_gs[0]
        dl_action['location'] = {'lat': next_gs[2], 'lon': next_gs[3]}
        dl_images = math.floor(dl_time / settings['dl-steps-per-image'])

        # 3. Determine next state: storage, curr_time
        if dl_images == 0 or len(sat['storage']) == 0:
            dl_action['reward'] = 0.0
            dl_action['storage'] = deepcopy(sat['storage'])
            dl_action['soonest'] = math.ceil(next_gs[1])
            # print('--> NO IMAGES COULD BE DOWNLINKED')
            return [dl_action]

        dl_size = min(dl_images, len(sat['storage']))
        for dl_img in range(dl_size):
            img = dl_action['storage'].pop(0)
            dl_action['reward'] += settings['dl-rewards'][img]

        dl_end = dl_start + dl_size * settings['dl-steps-per-image']
        dl_action['soonest'] = dl_end

        # 4. Large penalty if down-linking without full storage
        if len(sat['storage']) != settings['sat-storage-cap']:
            free_storage = abs(settings['sat-storage-cap'] - len(sat['storage']))
            dl_action['reward'] += (settings['dl-free-storage-penalty'] * free_storage)

        # print('--> DOWNLINKED IMAGES', dl_size, dl_end, sat['storage'])

        return [dl_action]





















