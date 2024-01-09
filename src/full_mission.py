import datetime
import os

from create_mission import create_mission
from execute_mission import execute_mission
from process_mission import process_mission
from plan_mission import plan_mission
from plot_mission_cartopy import plot_mission

def main():
    cross_track_ffor = 15 # deg
    along_track_ffor = 15 # deg
    cross_track_ffov = 0 # deg
    along_track_ffov = 0 # deg
    agility = 1 # deg/s
    num_planes = 1
    num_sats_per_plane = 1
    settings = {
        "directory": "./missions/landsat_rain/",
        "step_size": 10,
        "duration": 1,
        "plot_interval": 10,
        "plot_duration": 0.2,
        "plot_location": "./missions/landsat_rain/plots/",
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "preplanned_observations": None,
        "event_csvs": ["./rain_events.csv"],
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
        "reobserve_reward": 0.1,
        "sharing_horizon": 1000,
        "planning_horizon": 1000
    }
    # if not os.path.exists(settings["directory"]):
    #     os.mkdir(settings["directory"])
    # if not os.path.exists(settings["directory"]+'orbit_data/'):
    #     os.mkdir(settings["directory"]+'orbit_data/')
    # create_mission(settings)
    # execute_mission(settings)
    # # if settings["preplanned_observations"] is None:
    # #     plan_mission(settings) # must come before process as process expects a plan.csv in the orbit_data directory
    # process_mission(settings)
    plot_mission(settings)


if __name__ == "__main__":
    main()