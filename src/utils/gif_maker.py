import numpy as np
import imageio.v2 as iio
import datetime

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
    "plot_location": "/home/ben/repos/satplan/missions/chrissi_results/"+f,
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
start_frac = 0
num_skip = settings["plot_interval"]
steps = np.arange(int(np.floor(settings["duration"]*start_frac*86400/settings["step_size"])),int(np.floor(settings["duration"]*86400*settings["plot_duration"]/settings["step_size"])),num_skip)
filenames = []
for step in steps:
    filenames.append(f'{settings["plot_location"]}/frame_{step}.png')
print(filenames)
# print('Charts saved\n')
gif_name = settings["directory"]+'animation'
# Build GIF
print('Creating gif\n')
with iio.get_writer(f'{gif_name}.gif', mode='I') as writer:
    for filename in filenames:
        image = iio.imread(filename)
        writer.append_data(image)
print('Gif saved\n')