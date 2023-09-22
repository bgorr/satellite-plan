import numpy as np
import datetime
import csv
import os
import multiprocessing
from functools import partial
from tqdm import *

def unique(lakes):
    lakes = np.asarray(lakes)
    return np.unique(lakes,axis=0)

def close_enough(lat0,lon0,lat1,lon1):
    if np.sqrt((lat0-lat1)**2+(lon0-lon1)**2) < 0.01:
        return True
    else:
        return False

def get_step_data(step_num,settings):
    vis_rows = []
    with open(settings["directory"]+'sat_visibilities/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            vis_rows.append(row)

    obs_rows = []
    with open(settings["directory"]+'sat_observations/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            obs_rows.append(row)

    event_rows = []
    with open(settings["directory"]+'events/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            event_rows.append(row)
    event_obs_count = 0
    event_obs_list = []
    event_obs_pair_list = []
    event_vis_count = 0
    event_vis_list = []
    event_vis_pair_list = []
    for event in event_rows:
        for obs in obs_rows:
            if close_enough(float(obs[2]),float(obs[3]),float(event[0]),float(event[1])):
                event_obs_count += 1
                event_obs_pair = {
                    "event": event,
                    "obs": obs,
                    "step": step_num
                }
                event_obs_list.append(event)
                event_obs_pair_list.append(event_obs_pair)
        for vis in vis_rows:
            if close_enough(float(vis[2]),float(vis[3]),float(event[0]),float(event[1])):
                event_vis_count += 1
                event_vis_pair = {
                    "event": event,
                    "vis": vis,
                    "step": step_num
                }
                event_vis_list.append(event)
                event_vis_pair_list.append(event_vis_pair)

    results = {
        "event_obs_count": event_obs_count,
        "event_obs_list": event_obs_list,
        "event_obs_pair_list": event_obs_pair_list,
        "event_vis_count": event_vis_count,
        "event_vis_list": event_vis_list,
        "event_vis_pair_list": event_vis_pair_list
    }
    return results

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
event_duration = 7200 # second
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

pool = multiprocessing.Pool()
steps = np.arange(0,int(np.floor(settings["duration"]*86400/settings["step_size"])),1)
print(steps)
max_ = len(steps)
result_list = []
with tqdm(total=max_) as pbar:
    for result in pool.imap_unordered(partial(get_step_data, settings=settings), steps, chunksize=5):
        pbar.update()
        result_list.append(result)
#result_list = list(tqdm(pool.imap(partial(get_step_data, settings=settings), steps)))
cumulative_event_obs_list = []
cumulative_event_vis_list = []
event_obs_count_sum = 0
event_vis_count_sum = 0
for result in result_list:
    event_obs_count_sum += result["event_obs_count"]
    event_vis_count_sum += result["event_vis_count"]
    cumulative_event_obs_list.extend(result["event_obs_list"])
    cumulative_event_vis_list.extend(result["event_vis_list"])

print("Number of event observations: "+str(event_obs_count_sum))
print("Number of event visibilities: "+str(event_vis_count_sum))
print("Number of events observed: "+str(unique(cumulative_event_obs_list)))
print("Number of events visible: "+str(unique(cumulative_event_vis_list)))