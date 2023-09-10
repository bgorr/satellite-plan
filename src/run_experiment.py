import datetime
import os
import numpy as np
import csv

from create_mission import create_mission
from execute_mission import execute_mission
from process_mission import process_mission
from plan_mission import plan_mission
from plot_mission_cartopy import plot_mission

def main():
    mission_name = "experiment1"
    cross_track_ffor = 60 # deg
    along_track_ffor = 60 # deg
    cross_track_ffov = 10 # deg
    along_track_ffov = 10 # deg
    agility = 1 # deg/s
    num_planes = 5 
    num_sats_per_plane = 5
    var = 10 # deg lat/lon
    num_points_per_cell = 20
    simulation_step_size = 10 # seconds
    simulation_duration = 1 # days
    event_frequency = 1e-4 # events per second
    event_duration = 7200 # seconds
    steps = np.arange(0,simulation_duration*86400,simulation_step_size)

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
        "plot_clouds": False,
        "plot_rain": False,
        "plot_obs": True,
        "plot_duration": 1,
        "plot_interval": 20,
        "plot_location": "./missions/"+mission_name+"/plots/",
        "cross_track_ffor": cross_track_ffor,
        "along_track_ffor": along_track_ffor,
        "cross_track_ffov": cross_track_ffov,
        "along_track_ffov": along_track_ffov,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "process_obs_only": False
    }
    if not os.path.exists(settings["directory"]):
        os.mkdir(settings["directory"])
    if not os.path.exists(settings["directory"]+'orbit_data/'):
        os.mkdir(settings["directory"]+'orbit_data/')
    create_mission(settings)
    execute_mission(settings)
    if settings["preplanned_observations"] is None:
        plan_mission(settings) # must come before process as process expects a plan.csv in the orbit_data directory
    process_mission(settings)
    plot_mission(settings)


if __name__ == "__main__":
    main()