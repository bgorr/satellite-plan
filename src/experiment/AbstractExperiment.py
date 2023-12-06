import numpy as np
import os
import config
import csv
from copy import deepcopy
import datetime
import json
import re
import time
from multiprocessing import Pool
from tqdm import tqdm
from planners import utils

from create_mission import create_mission
from execute_mission import execute_mission
from process_mission import process_mission
from planners.BaseRL import BaseRL
from planners.TransformerRL import TransformerRL
from plot_mission_cartopy import plot_mission
from planners.utils import record_results, record_json_results

from utils.convert_geo import convert_geo_cords
from results.ExperimentResult import ExperimentResult

from planners.AbstractPlanner import AbstractPlanner
from planners.FifoPlanner import FifoPlanner
from planners.VDNPlanner import VDNPlanner
from planners.VDNPlannerTrans import VDNPlannerTrans
from planners.PPOPlanner import PPOPlanner

"""

0.2 interp

*** RND ***
-----> PLANNED OBS: 1092 / 9436
--------> SEEN OBS: 1790 / 9436
-----> EVENTS SEEN: 94
--> AVG EVENT SEEN: 0.05964467005076142


*** FIFO ***
-----> PLANNED OBS: 1719 / 9436
--------> SEEN OBS: 2301 / 9436
-----> EVENTS SEEN: 130
--> AVG EVENT SEEN: 0.08248730964467005

-----> PLANNED OBS: 1719 / 9436
--------> SEEN OBS: 2249 / 9436
-----> EVENTS SEEN: 126
--> AVG EVENT SEEN: 0.0799492385786802




"""

default_settings = {
    "name": "def-1day-3sat",  # rl_default_3day
    "planner": "vdn",
    "episodes": 1000,
    "ffor": 60,
    "ffov": 5,
    "cross_track_ffor": 60,  # ffor
    "along_track_ffor": 60,  # ffor
    "cross_track_ffov": 5,   # ffov
    "along_track_ffov": 5,   # ffov
    "duration": 1,  # days
    "initial_datetime": datetime.datetime(2020, 1, 1, 0, 0, 0),
    "agility": 0.5,
    "event_duration": 6 * 3600,  # 6 hours event duration (seconds)
    "event_frequency": 0.01 / 3600,  # probability event gets created at each location per time-step
    "event_density": 10,  # points considered per 10 deg lat/lon grid cell
    "event_clustering": 4,  # specifies clustering of points in lat/lon grid cells (var of gaussian dist)
    "planner_options": {
        "reobserve": "encouraged",
        "reobserve_reward": 1
    },
    "reward": 0,
    "step_size": 10,  # seconds
    "grid_type": "event",
    "process_obs_only": False,

    # Constellation Params
    "constellation_size": 3,
    "num_planes": 3,
    "num_sats_per_plane": 1,

    # Satellite Params
    'sat-storage-cap': 3,
    'dl-steps-per-image': 1,  # GS Accesses windows typically 40-50 steps
    'mem-overflow-penalty': -5,

    # Downlink Rewards
    'dl-obs-reward': 0.0,
    'dl-event-reward': 1,
    'dl-nothing-reward': 0.0,
    'dl-rewards': {'N': 0.0, 'E': 1.0, 'NE': 1.0},
    'dl-free-storage-penalty': 0.0,

    # Observation Rewards
    'obs-novel-event-reward': 1.0,
    'obs-event-reward': 0.0,
    'obs-reward': 0,
}


class AbstractExperiment:

    def __init__(self, settings=None):
        self.settings = default_settings
        if settings:
            self.settings = settings
        self.settings['directory'] = os.path.join(config.root_dir, 'missions', self.settings['name'])
        self.settings['steps'] = self.init_steps()
        self.settings['event_locations'], self.settings['grid_points'] = self.init_event_locations()
        self.settings['events'] = self.init_events()
        self.settings['orbit_data'] = self.init_orbit_data()
        self.init_ground_stations()
        self.init_satellite_links()

        # Planner
        if self.settings['planner'] == 'random':
            self.planner = AbstractPlanner(self.settings)
        elif self.settings['planner'] == 'fifo':
            self.planner = FifoPlanner(self.settings)
        elif self.settings['planner'] == 'vdn':
            self.planner = VDNPlanner(self.settings)
        elif self.settings['planner'] == 'vdn-trans':
            self.planner = VDNPlannerTrans(self.settings)
        elif self.settings['planner'] == 'ppo':
            self.planner = PPOPlanner(self.settings)
        else:
            raise ValueError("--> INVALID PLANNER TYPE")


    # ---------------------------------------------
    # Initialize
    # ---------------------------------------------

    def init_steps(self):
        return np.arange(0, self.settings['duration'] * 86400, self.settings['step_size'])

    def init_event_locations(self):
        # Grid Event Locations
        events_file_dir = os.path.join(config.root_dir, 'coverage_grids', self.settings['name'])
        events_file_path = os.path.join(config.root_dir, 'coverage_grids', self.settings['name'], 'event_locations.csv')
        self.settings['point_grid'] = events_file_path
        event_locations = []
        grid_points = []
        if not os.path.exists(events_file_path):
            center_lats = np.arange(-85, 95, 10)
            center_lons = np.arange(-175, 185, 10)
            clustering = self.settings['event_clustering']
            density = self.settings["event_density"]
            for clat in center_lats:
                for clon in center_lons:
                    mean = [clat, clon]
                    cov = [[clustering, 0], [0, clustering]]
                    xs, ys = np.random.multivariate_normal(mean, cov, density).T
                    for i in range(len(xs)):
                        location = [xs[i], ys[i]]
                        event_locations.append(location)
                        grid_points.append({
                            'location': {
                                'lat': xs[i],
                                'lon': ys[i]
                            },
                            'times_observed': 0,
                        })
            if not os.path.exists(events_file_dir):
                os.mkdir(events_file_dir)
            with open(events_file_path, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['lat [deg]', 'lon [deg]'])
                for location in event_locations:
                    csvwriter.writerow(location)
        else:
            with open(self.settings["point_grid"], 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvfile)
                for row in csvreader:
                    event_locations.append([float(row[0]), float(row[1])])
                    grid_points.append({
                        'location': {
                            'lat': float(row[0]),
                            'lon': float(row[1])
                        },
                        'times_observed': 0,
                        'events': [],
                    })
        return event_locations, grid_points
        
    def init_events(self):
        event_frequency = self.settings["event_frequency"]
        event_duration = self.settings["event_duration"]
        events_file_dir_2 = os.path.join(config.root_dir, 'events', self.settings['name'])
        events_file_path_2 = os.path.join(config.root_dir, 'events', self.settings['name'], 'events.csv')
        self.settings['event_csvs'] = [events_file_path_2]
        active_events = []
        if not os.path.exists(events_file_path_2):
            for step in self.settings['steps']:
                for location in self.settings['event_locations']:  # event lat / lon pair
                    if np.random.random() < event_frequency * self.settings['step_size']:
                        event = [location[0], location[1], step, event_duration, 1]
                        # event = [location[0], location[1], 0, self.settings['duration'] * 24 * 3600, 1]  # Event lasts as long as mission
                        active_events.append(event)
            if not os.path.exists(events_file_dir_2):
                os.mkdir(events_file_dir_2)
            with open(events_file_path_2, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['lat [deg]', 'lon [deg]', 'start time [s]', 'duration [s]', 'severity'])
                for event in active_events:
                    csvwriter.writerow(event)

        events = []
        with open(events_file_path_2, newline='') as csv_file:
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
                    "start": float(row[2]) / self.settings["step_size"],                  # steps
                    "end": (float(row[2]) + float(row[3])) / self.settings["step_size"],  # steps
                    "severity": float(row[4]),
                    "times_observed": 0,
                    "rarity": 1,
                    "reward": 1,
                }
                events.append(event)
                for grid_point in self.settings['grid_points']:
                    gp_lat = grid_point['location']['lat']
                    gp_lon = grid_point['location']['lon']
                    if gp_lat == event['location']['lat'] and gp_lon == event['location']['lon']:
                        grid_point['events'].append(event)
                        break
        return events

    def init_orbit_data(self):
        orbit_data_dir = os.path.join(self.settings['directory'], 'orbit_data')
        if not os.path.exists(orbit_data_dir):
            os.makedirs(orbit_data_dir)
            create_mission(self.settings)
            execute_mission(self.settings)
        convert_geo_cords(self.settings)

        orbit_data = {}
        for subdir_idx, subdir in enumerate(os.listdir(orbit_data_dir)):
            if "comm" in subdir:
                continue
            if ".json" in subdir:
                continue
            sat_files = os.listdir(os.path.join(orbit_data_dir, subdir))
            if subdir not in orbit_data:
                orbit_data[subdir] = {
                    "orbitpy_idx": subdir,
                    'comm_windows': [],
                    'gs_time_step_windows': [],
                }
            for f in sat_files:
                if "datametrics" in f:
                    datametrics_path = os.path.join(orbit_data_dir, subdir, f)
                    with open(datametrics_path, newline='') as csv_file:
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
                    orbit_data[subdir]["visibilities"] = visibilities

                if "state_geo" in f:
                    state_geo_path = os.path.join(orbit_data_dir, subdir, f)
                    with open(state_geo_path, newline='') as csv_file:
                        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                        nadir_lat_lons = []
                        for idx, row in enumerate(spamreader):
                            if idx < 1:
                                continue
                            row = [float(i) for i in row]
                            nadir_lat_lons.append([row[0], row[1], row[2]])
                    orbit_data[subdir]["nadir_lat_lons"] = nadir_lat_lons

                if "state_cartesian" in f:
                    state_cartesian_path = os.path.join(orbit_data_dir, subdir, f)
                    with open(state_cartesian_path, newline='') as csv_file:
                        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                        states = []
                        i = 0
                        for row in spamreader:
                            if i < 5:
                                i = i + 1
                                continue
                            row = [float(i) for i in row]
                            states.append(row)
                    orbit_data[subdir]["orbitpy_id"] = subdir
                    orbit_data[subdir]["states"] = states




        orbits_processed = []
        all_visibilities = []
        for key, value in orbit_data.items():
            orbits_processed.append(key)
            all_visibilities.append(value['visibilities'])

        with Pool(processes=config.cores) as pool:
            all_windows = list(tqdm(pool.imap(utils.init_sat_observations_proc, all_visibilities), total=len(all_visibilities)))
        orbit_windows = [window['windows'] for window in all_windows]
        orbit_viz_windows = [window['viz_windows'] for window in all_windows]
        for idx, sat in enumerate(orbits_processed):
            orbit_data[sat]['all_obs'] = orbit_windows[idx]
            orbit_data[sat]['all_viz'] = orbit_viz_windows[idx]

        return orbit_data

    def init_ground_stations(self):
        # for idx, gs in enumerate(ground_stations):
        # iterate over satellite dirs
        orbit_data_dir = os.path.join(self.settings['directory'], 'orbit_data')
        mission_specs = json.load(open(os.path.join(orbit_data_dir, 'MissionSpecs.json')))
        ground_stations = mission_specs['groundStation']
        for subdir_idx, subdir in enumerate(os.listdir(orbit_data_dir)):
            if "comm" in subdir:
                continue
            if ".json" in subdir:
                continue
            sat_files = os.listdir(os.path.join(orbit_data_dir, subdir))
            for f in sat_files:
                if "gndStn" in f:
                    g_station_coords = [0.0, 0.0]
                    for idx, gs in enumerate(ground_stations):
                        if str(idx) in f:
                            g_station_coords = [float(gs['latitude']), float(gs['longitude'])]
                            break

                    gndStn_path = os.path.join(orbit_data_dir, subdir, f)
                    with open(gndStn_path, newline='') as csv_file:
                        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                        time_step_windows = []
                        i = 0
                        for row in spamreader:
                            if i < 4:
                                i = i + 1
                                continue
                            row = [float(i) for i in row]
                            for coord in g_station_coords:
                                row.append(coord)
                            time_step_windows.append(row)
                    if "gs_time_step_windows" not in self.settings['orbit_data'][subdir]:
                        self.settings['orbit_data'][subdir]['gs_time_step_windows'] = []
                    self.settings['orbit_data'][subdir]["gs_time_step_windows"].extend(time_step_windows)

    def init_satellite_links(self):
        orbit_data_dir = os.path.join(self.settings['directory'], 'orbit_data')
        for subdir_idx, subdir in enumerate(os.listdir(orbit_data_dir)):
            if "comm" in subdir:
                comm_files = os.listdir(os.path.join(orbit_data_dir, subdir))
                for f in comm_files:
                    match = re.search(r'sat(\d+)_to_sat(\d+).csv', f)
                    if match:
                        sat1 = 'sat' + match.group(1)
                        sat2 = 'sat' + match.group(2)
                        if sat1 not in self.settings['orbit_data'] or sat2 not in self.settings['orbit_data']:
                            raise ValueError('--> Error: Satellite not found in orbit data', sat1, sat2, self.settings['orbit_data'].keys())

                        comm_file_path = os.path.join(orbit_data_dir, subdir, f)
                        with open(comm_file_path, newline='') as csv_file:
                            spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                            sat1_comm_windows = []
                            sat2_comm_windows = []
                            i = 0
                            for row in spamreader:
                                if i < 4:
                                    i = i + 1
                                    continue
                                window = [float(i) for i in row]
                                sat1_window = deepcopy(window).append(int(sat2))
                                sat2_window = deepcopy(window).append(int(sat1))
                                sat1_comm_windows.append(sat1_window)
                                sat2_comm_windows.append(sat2_window)

                        if "comm_windows" not in self.settings['orbit_data'][sat1]:
                            self.settings['orbit_data'][sat1]['comm_windows'] = []
                        if "comm_windows" not in self.settings['orbit_data'][sat2]:
                            self.settings['orbit_data'][sat2]['comm_windows'] = []

                        self.settings['orbit_data'][sat1]["comm_windows"].extend(sat1_comm_windows)
                        self.settings['orbit_data'][sat2]["comm_windows"].extend(sat2_comm_windows)

    # ---------------------------------------------
    # Run
    # ---------------------------------------------

    def run(self):
        for episode in range(self.settings['episodes']):
            satellites = self.planner.train_episode()
            self.process_results(satellites)

    # ---------------------------------------------
    # Process Results
    # ---------------------------------------------

    def process_results(self, satellites):
        possible_obs = 0
        planned_obs = 0
        for sat in satellites:
            possible_obs += len(sat['all_obs'])
            planned_obs += len(sat['planned_obs'])
        points_seen = np.sum([point['times_observed'] for point in self.settings['grid_points']])


        events_seen_list = []
        for gp in self.settings['grid_points']:
            for event in gp['events']:
                events_seen_list.append(event['times_observed'])

        # print('-----> PLANNED OBS:', planned_obs, '/', possible_obs)
        # print('--------> SEEN OBS:', points_seen, '/', possible_obs)
        # print('-----> EVENTS SEEN:', np.sum(events_seen_list))
        # print('--> AVG EVENT SEEN:', np.mean(events_seen_list))







if __name__ == '__main__':
    experiment = AbstractExperiment()
    experiment.run()








