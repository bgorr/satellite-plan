import datetime
import os
import sys
sys.path.append(".")

from src.create_mission import create_mission
from src.execute_mission import execute_mission
from src.process_mission import process_mission
from src.plan_mission import plan_mission, plan_mission_replan_interval, plan_mission_replan_interval_het
from src.plot_mission import plot_mission
from src.plot_mission_heterogeneous import plot_mission_het
from src.utils.compute_experiment_statistics import compute_experiment_statistics
from src.utils.compute_experiment_statistics_het import compute_experiment_statistics_het
from src.utils.process_coobs import process_coobs

def main(homhet_flag):
    name = "full_mission_test_het"
    settings = {
        "name": name,
        "instrument": {
            "ffor": 30,
            "ffov": 0
        },
        "agility": {
            "slew_constraint": "rate",
            "max_slew_rate": 0.1,
            "inertia": 2.66,
            "max_torque": 4e-3
        },
        "orbit": {
            "altitude": 705, # km
            "inclination": 98.4, # deg
            "eccentricity": 0.0001,
            "argper": 0, # deg
        },
        "constellation": {
            "num_sats_per_plane": 3,
            "num_planes": 3,
            "phasing_parameter": 1
        },
        "events": {
            "event_duration": 3600*6,
            "event_frequency": 0.01/3600,
            "event_density": 2,
            "event_clustering": 4
        },
        "time": {
            "step_size": 10, # seconds
            "duration": 0.1, # days
            "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
        },
        "rewards": {
            "reward": 10,
            "reward_increment": 0.1,
            "reobserve_reward": 2
        },
        "plotting":{
            "plot_clouds": False,
            "plot_rain": False,
            "plot_duration": 0.1,
            "plot_interval": 10,
            "plot_obs": True
        },
        "planner": "dp",
        "event_csvs": [],
        "num_meas_types": 3,
        "sharing_horizon": 1000,
        "planning_horizon": 1000,
        "directory": "./missions/"+name+"/",
        "grid_type": "uniform", # can be "uniform" or "custom"
        "preplanned_observations": None,
        "process_obs_only": False,
        "conops": "onboard_processing"
    }
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
    create_mission(settings)
    execute_mission(settings)
    if homhet_flag == "homogeneous":
        if settings["preplanned_observations"] is None:
            plan_mission(settings) # must come before process as process expects a plan.csv in the orbit_data directory
        process_mission(settings)
        plot_mission(settings)
    elif homhet_flag == "heterogeneous":
        if settings["preplanned_observations"] is None:
            plan_mission(settings) # must come before process as process expects a plan.csv in the orbit_data directory
        process_mission(settings)
        plot_mission_het(settings)
    else:
        print("Invalid homhet_flag")


if __name__ == "__main__":
    main(homhet_flag="heterogeneous")