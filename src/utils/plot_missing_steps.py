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
    cross_track_ffor = 60 # deg
    along_track_ffor = 2 # deg
    cross_track_ffov = 0 # deg
    along_track_ffov = 0 # deg
    agility = 1 # deg/s
    num_planes = 1
    num_sats_per_plane = 5
    f = 'accesses_12h_rew_5sat_sol_5degs'
    settings = {
        "directory": "./missions/test_mission_6/",
        "step_size": 1,
        "duration": 1,
        "plot_interval": 5,
        "plot_duration": 2/24,
        "plot_location": "./missions/chrissi_results/"+f,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "static", # can be "event" or "static"
        "preplanned_observations": "./missions/test_mission_6/planner_outputs/accesses_2h_rew_5sat_sol_2degs.csv",
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
        "process_obs_only": False
    }
    plot_missing(settings)