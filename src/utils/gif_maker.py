import numpy as np
import imageio
import datetime

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
start_frac = 0
num_skip = settings["plot_interval"]
steps = np.arange(int(np.floor(settings["duration"]*start_frac*86400/settings["step_size"])),int(np.floor(settings["duration"]*86400*settings["plot_duration"]/settings["step_size"])),num_skip)
filenames = []
for step in steps:
    filenames.append(f'{settings["directory"]}plots/frame_{step}.png')
# print('Charts saved\n')
gif_name = settings["directory"]+'animation'
# Build GIF
print('Creating gif\n')
with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
print('Gif saved\n')