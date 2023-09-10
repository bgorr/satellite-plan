import os
import datetime

from src.process_mission import process_mission
from src.plot_mission_cartopy import plot_mission

plan_directory = './missions/test_mission_6/planner_outputs/'

for f in os.listdir(plan_directory):
    cross_track_ffor = 60 # deg
    along_track_ffor = 2 # deg
    cross_track_ffov = 0 # deg
    along_track_ffov = 0 # deg
    agility = 1 # deg/s
    num_planes = 1
    num_sats_per_plane = 5
    settings = {
        "directory": "./missions/test_mission_6/",
        "step_size": 1,
        "duration": 1,
        "plot_interval": 30,
        "plot_duration": 4/24,
        "plot_location": "./missions/chrissi_results/"+f[:-4],
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "static", # can be "event" or "static"
        "preplanned_observations": "./missions/test_mission_6/planner_outputs/"+f,
        "event_csvs": [],
        "plot_clouds": False,
        "plot_rain": False,
        "plot_obs": True,
        "cross_track_ffor": cross_track_ffor,
        "along_track_ffor": along_track_ffor,
        "cross_track_ffov": cross_track_ffov,
        "along_track_ffov": along_track_ffov,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "process_obs_only": True
    }
    if not os.path.exists("./missions/chrissi_results/"+f[:-4]):
        os.mkdir("./missions/chrissi_results/"+f[:-4])
        process_mission(settings)
        plot_mission(settings)