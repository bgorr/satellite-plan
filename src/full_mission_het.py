import datetime
import os

from create_mission_het import create_mission_het
from execute_mission import execute_mission
from process_mission import process_mission
from plan_mission import plan_mission, plan_mission_replan_interval, plan_mission_replan_interval_het
from plot_mission_cartopy_heterogeneous import plot_mission
from utils.compute_experiment_statistics_het import compute_experiment_statistics_het
from utils.process_coobs import process_coobs

def main():
    cross_track_ffor = 60 # deg
    along_track_ffor = 60 # deg
    cross_track_ffov = 0 # deg
    along_track_ffov = 0 # deg
    agility = 1 # deg/s
    num_planes = 6
    num_sats_per_plane = 6
    settings = {
        "directory": "./missions/agu_rain_het/",
        "step_size": 10,
        "duration": 1,
        "plot_interval": 100,
        "plot_duration": 1,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "preplanned_observations": None,
        "event_csvs": ["./rain_events_het.csv"],
        "point_grid": "./coverage_grids/agu_rain/event_locations.csv",
        "plot_clouds": False,
        "plot_rain": True,
        "plot_obs": True,
        "cross_track_ffor": cross_track_ffor,
        "along_track_ffor": along_track_ffor,
        "cross_track_ffov": cross_track_ffov,
        "along_track_ffov": along_track_ffov,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "planner": "dp",
        "process_obs_only": False,
        "reward": 10,
        "reward_increment": 0.1,
        "reobserve_reward": 2,
        "sharing_horizon": 1000,
        "planning_horizon": 1000,
        "experiment_settings": {
            "ffor": 60,
            "ffov": 0,
            "constellation_size": 6,
            "agility": 1,
            "event_duration": 4*3600,
            "event_frequency": 0.1/3600,
            "event_density": 10,
            "event_clustering": 16,
            "planner": "dp",
            "reobserve_reward": 2,
            "num_meas_types": 3,
            "reward": 10,
            "reward_increment": 0.1,
            "time_horizon": 1000
        }
    }
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
    # create_mission_het(settings)
    # execute_mission(settings)
    # if settings["preplanned_observations"] is None:
    #     plan_mission(settings) # must come before process as process expects a plan.csv in the orbit_data directory
    #     plan_mission_replan_interval(settings)
    #     plan_mission_replan_interval_het(settings)
    # process_mission(settings)
    overall_results = compute_experiment_statistics_het(settings)
    process_coobs(settings)
    plot_mission(settings)


if __name__ == "__main__":
    main()