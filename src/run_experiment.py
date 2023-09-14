import datetime
import os
import numpy as np
import csv

from create_mission import create_mission
from execute_mission import execute_mission
from process_mission import process_mission
from plan_mission import plan_mission, plan_mission_replan_interval
from plot_mission_cartopy import plot_mission
from utils.compute_experiment_statistics import compute_experiment_statistics

def run_experiment(experiment_settings):
    # mission_name = "experiment2"
    # cross_track_ffor = 60 # deg
    # along_track_ffor = 60 # deg
    # cross_track_ffov = 10 # deg
    # along_track_ffov = 10 # deg
    # agility = 1 # deg/s
    # num_planes = 4
    # num_sats_per_plane = 4
    # var = 5 # deg lat/lon
    # num_points_per_cell = 10
    simulation_step_size = 10 # seconds
    simulation_duration = 0.05 # days
    # event_frequency = 0.001/3600 # events per hour
    # event_duration = 3600*6 # second
    mission_name = experiment_settings["name"]
    cross_track_ffor = experiment_settings["ffor"]
    along_track_ffor = experiment_settings["ffor"]
    cross_track_ffov = experiment_settings["ffov"]
    along_track_ffov = experiment_settings["ffov"] # TODO carefully consider this assumption
    agility = experiment_settings["agility"]
    num_planes = experiment_settings["constellation_size"]
    num_sats_per_plane = experiment_settings["constellation_size"]
    var = experiment_settings["event_clustering"]
    num_points_per_cell = experiment_settings["event_density"]
    event_frequency = experiment_settings["event_frequency"]
    event_duration = experiment_settings["event_duration"]
    steps = np.arange(0,simulation_duration*86400,simulation_step_size)
    if not os.path.exists("./coverage_grids/"+mission_name+"/event_locations.csv"):
        event_locations = []
        center_lats = np.arange(-85,95,10)
        center_lons = np.arange(-175,185,10)
        for clat in center_lats:
            for clon in center_lons:
                mean = [clat, clon]
                cov = [[var, 0], [0, var]]
                xs, ys = np.random.multivariate_normal(mean, cov, num_points_per_cell).T
                for i in range(len(xs)):
                    location = [xs[i],ys[i]]
                    event_locations.append(location)
        if not os.path.exists("./coverage_grids/"+mission_name+"/"):
                os.mkdir("./coverage_grids/"+mission_name+"/")
        with open("./coverage_grids/"+mission_name+"/event_locations.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]'])
            for location in event_locations:
                csvwriter.writerow(location)
    
    if not os.path.exists("./events/"+mission_name+"/events.csv"):
        events = []
        for step in steps:
            for location in event_locations:
                if np.random.random() < event_frequency*simulation_step_size:
                    event = [location[0],location[1],step,event_duration,1]
                    events.append(event)
        if not os.path.exists("./events/"+mission_name+"/"):
            os.mkdir("./events/"+mission_name+"/")
        with open("./events/"+mission_name+"/events.csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity'])
            for event in events:
                csvwriter.writerow(event)

    settings = {
        "directory": "./missions/"+mission_name+"/",
        "step_size": simulation_step_size,
        "duration": simulation_duration,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "point_grid": "./coverage_grids/"+mission_name+"/event_locations.csv",
        "preplanned_observations": None,
        "event_csvs": ["./events/"+mission_name+"/events.csv"],
        "cross_track_ffor": cross_track_ffor,
        "along_track_ffor": along_track_ffor,
        "cross_track_ffov": cross_track_ffov,
        "along_track_ffov": along_track_ffov,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "process_obs_only": False,
        "planner": "fifo",
        "experiment_settings": experiment_settings
    }
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
    create_mission(settings)
    execute_mission(settings)
    #plan_mission(settings) # must come before process as process expects a plan.csv in the orbit_data directory
    #plan_mission_replan_interval(settings)
    compute_experiment_statistics(settings)


if __name__ == "__main__":
    run_experiment()