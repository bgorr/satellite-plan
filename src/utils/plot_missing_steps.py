import time, calendar, datetime
from functools import partial
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import urllib, os
import csv
import numpy as np
import imageio
import sys
import multiprocessing
import h5py
import matplotlib.colors as colors
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
from multiprocessing import set_start_method
from src.plot_mission_cartopy import plot_step

def plot_missing(settings):
    if not os.path.exists(settings["directory"]+'/'):
        os.mkdir(settings["directory"]+'/')
    # PLOTS THE LAST 1/4th OF THE SIMULATION
    # imageio gif creation kills itself if there are too many images, is there a fix or is it just a WSL issue?
    start_frac = 0
    num_skip = settings["plot_interval"]
    steps = np.arange(int(np.floor(settings["duration"]*start_frac*86400/settings["step_size"])),int(np.floor(settings["duration"]*86400*settings["plot_duration"]/settings["step_size"])),num_skip)
    for step in steps:
        if not os.path.exists(f'{settings["plot_location"]}/frame_{step}.png'):
            plot_step(step,settings)

if __name__ == "__main__":
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
    plot_missing(settings)