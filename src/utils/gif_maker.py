import numpy as np
import imageio
import datetime

settings = {
        "directory": "./missions/test_mission_6/",
        "step_size": 1,
        "duration": 1,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "static", # can be "event" or "static"
        "preplanned_observations": "./missions/test_mission_6/planner_outputs/accesses_2h_rew_5sat_sol_2degs.csv",
        "event_csvs": [],
        "plot_clouds": False,
        "plot_rain": False,
        "plot_obs": True
    }
start_frac = 0
num_skip = 100
steps = np.arange(int(np.floor(settings["duration"]*start_frac*86400/settings["step_size"])),int(np.floor(settings["duration"]*86400/settings["step_size"])),num_skip)
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