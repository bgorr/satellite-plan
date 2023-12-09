import numpy as np
import datetime
import csv
import os
  

def process_coobs(settings):
    print("Processing coobs")
    base_directory = settings["directory"]

    directory = base_directory+"orbit_data/"

    with open(base_directory+"coobs_obs.csv",newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        coobs = []
        for row in spamreader:
            row = [float(i) for i in row[0:6]]
            coobs.append(row)

    timestep = settings["step_size"]
    duration = settings["duration"]*86400
    steps = np.arange(0,duration,timestep,dtype=int)
    if not os.path.exists(base_directory+'coobs'):
        os.mkdir(base_directory+'coobs')
    past_observations = []
    for i in range(len(steps)-1):      
        for observation in coobs:
            if steps[i] < observation[3] < steps[i+1]:
                prev_obs = None
                for past_obs in past_observations:
                    if past_obs[1] == observation[0]:
                        prev_obs = past_obs
                if prev_obs is not None:
                    new_observation = [prev_obs[0]+1,observation[0],observation[1]]
                    past_observations.remove(prev_obs)
                else:
                    new_observation = [1,observation[0],observation[1]]
                past_observations.append(new_observation)
        with open(base_directory+'coobs/step'+str(i)+'.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for obs in past_observations:
                csvwriter.writerow(obs)

    print("Processed coobs!")

if __name__ == "__main__":
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
        "plot_interval": 20,
        "plot_duration": 6/24,
        "plot_location": ".",
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
    process_coobs(settings)