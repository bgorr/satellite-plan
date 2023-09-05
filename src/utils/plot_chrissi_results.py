import os
import datetime

from src.process_mission import process_mission
from src.plot_mission_cartopy import plot_mission

plan_directory = './missions/test_mission_6/planner_outputs/'

for f in os.listdir(plan_directory):
    settings = {
        "directory": "./missions/test_mission_6/",
        "step_size": 1,
        "duration": 1,
        "plot_interval": 5,
        "plot_duration": 2/24,
        "plot_location": "./missions/chrissi_results/"+f[:-4],
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "static", # can be "event" or "static"
        "preplanned_observations": plan_directory+f,
        "event_csvs": [],
        "plot_clouds": False,
        "plot_rain": False,
        "plot_obs": True,
        "process_obs_only": True
    }
    os.mkdir("./missions/chrissi_results/"+f[:-4])
    process_mission(settings)
    plot_mission(settings)