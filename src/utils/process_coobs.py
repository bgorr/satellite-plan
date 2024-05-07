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

    timestep = settings["time"]["step_size"]
    duration = settings["time"]["duration"]*86400
    steps = np.arange(0,duration,timestep,dtype=int)
    if not os.path.exists(base_directory+'coobs'):
        os.mkdir(base_directory+'coobs')
    past_observations = []
    for i in range(len(steps)-1):
        for observation in coobs:
            if steps[i] <= observation[3]*timestep < steps[i+1]:
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
    name = "flood_grid_search_het_1"
    settings = {
        "name": name,
        "instrument": {
            "ffor": 60,
            "ffov": 5
        },
        "agility": {
            "slew_constraint": "rate",
            "max_slew_rate": 1,
            "inertia": 2.66,
            "max_torque": 4e-3
        },
        "orbit": {
            "altitude": 705, # km
            "inclination": 98.4, # deg
            "eccentricity": 0.0001,
            "argper": 0, # deg
        },
        "constellation": {
            "num_sats_per_plane": 8,
            "num_planes": 3,
            "phasing_parameter": 1
        },
        "events": {
            "event_duration": 5000,
            "num_events": 100,
            "event_clustering": "clustered"
        },
        "time": {
            "step_size": 10, # seconds
            "duration": 1, # days
            "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
        },
        "rewards": {
            "reward": 10,
            "reward_increment": 1,
            "reobserve_conops": "no_change",
            "event_duration_decay": "step",
            "no_event_reward": 5,
            "oracle_reobs": "true",
            "initial_reward": 5
        },
        "planner": "dp",
        "num_meas_types": 2,
        "sharing_horizon": 100,
        "planning_horizon": 5000,
        "directory": "./missions/"+name+"/",
        "grid_type": "custom", # can be "uniform" or "custom"
        "point_grid": "./missions/"+name+"/coverage_grids/event_locations.csv",
        "preplanned_observations": None,
        "event_csvs": ["./missions/"+name+"/events/events.csv"],
        "process_obs_only": False,
        "conops": "onboard_processing"
    }
    process_coobs(settings)