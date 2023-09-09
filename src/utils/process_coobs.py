import numpy as np
import datetime
import csv
import os
  

def process_mission(settings):
    print("Processing coobs")
    base_directory = settings["directory"]

    directory = base_directory+"orbit_data/"

    with open("./missions/test_mission_5/coobs_obs.csv",newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        coobs = []
        for row in spamreader:
            row = [float(i) for i in row]
            coobs.append(row)

    timestep = settings["step_size"]
    duration = settings["duration"]*86400
    steps = np.arange(0,duration,timestep,dtype=int)
    if not os.path.exists(base_directory+'coobs'):
        os.mkdir(base_directory+'coobs')
    print(coobs)
    past_observations = []
    for i in range(len(steps)):      
        for observation in coobs:
            if observation[2] == i:
                prev_obs = None
                for past_obs in past_observations:
                    if past_obs[1] == observation[0]:
                        prev_obs = past_obs
                if prev_obs is not None:
                    new_observation = [prev_obs[0]+1,observation[0],observation[1],observation[5]]
                    past_observations.remove(prev_obs)
                else:
                    new_observation = [1,observation[0],observation[1],observation[5]]
                past_observations.append(new_observation)
        with open(base_directory+'coobs/step'+str(i)+'.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for obs in past_observations:
                csvwriter.writerow(obs)

    print("Processed coobs!")

if __name__ == "__main__":
    settings = {
        "directory": "./missions/test_mission_5/",
        "step_size": 10,
        "duration": 1,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "event_csvs": ['bloom_events.csv','level_events.csv','temperature_events.csv'],
        "plot_clouds": False,
        "plot_rain": False,
        "plot_obs": False,
        "process_obs_only": True
    }
    process_mission(settings)