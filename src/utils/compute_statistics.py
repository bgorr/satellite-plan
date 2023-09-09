import numpy as np
import datetime
import csv
import os
import multiprocessing
from functools import partial

def unique(lakes):
    lakes = np.asarray(lakes)[:,0:1]
    return np.unique(lakes,axis=0)

def close_enough(lat0,lon0,lat1,lon1):
    if np.sqrt((lat0-lat1)**2+(lon0-lon1)**2) < 0.01:
        return True
    else:
        return False

def chunks(xs, n):
    n = max(1, n)
    return (xs[i:i+n] for i in range(0, len(xs), n))

def compute_statistics_pieces(input):
    events = input["events"]
    observations = input["observations"]
    all_events_count = 0
    bloom_events_count = 0
    temp_events_count = 0
    level_events_count = 0
    all_events_reward = 0
    bloom_events_reward = 0
    temp_events_reward = 0
    level_events_reward = 0
    bloom_events = []
    temp_events = []
    level_events = []
    observed_locations = []
    step_size = 10

    for event in events:
        measurements_str : str = event[5]
        measurements_str = measurements_str.replace('[','')
        measurements_str = measurements_str.replace(']','')
        measurements_str = measurements_str.replace(' ','')
        measurements_str = measurements_str.replace('\'','')
        measurements = measurements_str.split(',')
        if len(measurements) == 3:
            event_type = "bloom"
        elif "thermal" in measurements:
            event_type = "temp"
        elif "sar" in measurements:
            event_type = "level" 
    
        for obs in observations:
            if obs[0] > ((float(event[2])+float(event[3])) + 6000):
                break
            if close_enough(obs[2],obs[3],float(event[0]),float(event[1])):
                if (float(event[2]) < obs[0] < (float(event[2])+float(event[3]))) or (float(event[2]) < obs[1] < (float(event[2])+float(event[3]))):
                    if satellite_name_dict[obs[4]] in measurements:
                        measurements.remove(satellite_name_dict[obs[4]])

        if len(measurements) == 0:
            print("whoa a co-obs")
            print(event)
            if event_type == "bloom":
                bloom_events_count += 1
                bloom_events_reward += float(event[4])
                all_events_count += 1
                all_events_reward += float(event[4])
                bloom_events.append(event)
            if event_type == "temp":
                temp_events_count += 1
                temp_events_reward += float(event[4])
                all_events_count += 1
                all_events_reward += float(event[4])
                temp_events.append(event)
            if event_type == "level":
                level_events_count += 1
                level_events_reward += float(event[4])
                all_events_count += 1
                all_events_reward += float(event[4])
                level_events.append(event) 
    for obs in observations:
        observed_locations.append([obs[2],obs[3]])
    output = {}
    output["all_events_count"] = all_events_count
    output["bloom_events_count"] = bloom_events_count
    output["temp_events_count"] = temp_events_count
    output["level_events_count"] = level_events_count
    output["all_events_reward"] = all_events_reward
    output["bloom_events_reward"] = bloom_events_reward
    output["temp_events_reward"] = temp_events_reward
    output["level_events_reward"] = level_events_reward
    output["num_observations"] = len(observed_locations)
    output["num_unique_observations"] = len(unique(observed_locations))
    return output

    
def compute_statistics(events,observations,file_descriptor,step_size):

    all_events_count = 0
    bloom_events_count = 0
    temp_events_count = 0
    level_events_count = 0
    all_events_reward = 0
    bloom_events_reward = 0
    temp_events_reward = 0
    level_events_reward = 0
    bloom_events = []
    temp_events = []
    level_events = []
    observed_locations = []

    for event in events:
        measurements_str : str = event[5]
        measurements_str = measurements_str.replace('[','')
        measurements_str = measurements_str.replace(']','')
        measurements_str = measurements_str.replace(' ','')
        measurements_str = measurements_str.replace('\'','')
        measurements = measurements_str.split(',')
        if len(measurements) == 3:
            event_type = "bloom"
        elif "thermal" in measurements:
            event_type = "temp"
        elif "sar" in measurements:
            event_type = "level"

        for obs in observations:
            if obs[0]*step_size > ((float(event[2])+float(event[3])) + 6000):
                break
            if close_enough(obs[2],obs[3],float(event[0]),float(event[1])):
                if (float(event[2]) < obs[0]*step_size < (float(event[2])+float(event[3]))) or (float(event[2]) < obs[1]*step_size < (float(event[2])+float(event[3]))):
                    if satellite_name_dict[obs[4]] in measurements:
                        measurements.remove(satellite_name_dict[obs[4]])

        if len(measurements) == 0:
            print("whoa a co-obs")
            print(event)
            if event_type == "bloom":
                bloom_events_count += 1
                bloom_events_reward += float(event[4])
                all_events_count += 1
                all_events_reward += float(event[4])
                bloom_events.append(event)
            if event_type == "temp":
                temp_events_count += 1
                temp_events_reward += float(event[4])
                all_events_count += 1
                all_events_reward += float(event[4])
                temp_events.append(event)
            if event_type == "level":
                level_events_count += 1
                level_events_reward += float(event[4])
                all_events_count += 1
                all_events_reward += float(event[4])
                level_events.append(event) 

    print("All events co-observed:"+str(all_events_count))
    print("Bloom events co-observed:"+str(bloom_events_count))
    print("Temperature events co-observed:"+str(temp_events_count))
    print("Level events co-observed:"+str(level_events_count))

    print("All events co-observed reward:"+str(all_events_reward))
    print("Bloom events co-observed reward:"+str(bloom_events_reward))
    print("Temperature events co-observed reward:"+str(temp_events_reward))
    print("Level events co-observed reward:"+str(level_events_reward))

    for obs in observations:
        observed_locations.append([obs[2],obs[3]])
    print("Number of observations: "+str(len(observed_locations)))
    print("Number of unique observation locations: "+str(len(unique(observed_locations))))

    with open('./missions/test_mission_5/coobs_plan_'+file_descriptor+'.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for event in bloom_events:
            bloom_event = event[:-1]
            bloom_event.append(0)
            csvwriter.writerow(bloom_event)
        for event in temp_events:
            temp_event = event[:-1]
            temp_event.append(1)
            csvwriter.writerow(temp_event)
        for event in level_events:
            level_event = event[:-1]
            level_event.append(2)
            csvwriter.writerow(level_event)

settings = {
    "directory": "./missions/test_mission_5/",
    "step_size": 10,
    "duration": 1,
    "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
    "grid_type": "event", # can be "event" or "static"
    "event_csvs": ['bloom_events.csv','level_events.csv','temperature_events.csv'],
    "preplanned_observations": None,
    "plot_clouds": False,
    "plot_rain": False,
    "plot_obs": False,
    "process_obs_only": True
}

directory = settings["directory"]+"orbit_data/"

satellites = []
all_observations = []
all_observations_replan = []
all_visibilities = []
satellite_name_dict = {
    "sat0": "visible",
    "sat1": "visible",
    "sat2": "visible",
    "sat3": "sar",
    "sat4": "sar",
    "sat5": "sar",
    "sat6": "thermal",
    "sat7": "thermal",
    "sat8": "thermal",
}


for subdir in os.listdir(directory):
    satellite = {}
    if "comm" in subdir:
        continue
    if ".json" in subdir:
        continue
    for f in os.listdir(directory+subdir):
        if "state_cartesian" in f:
            with open(directory+subdir+"/"+f,newline='') as csv_file:
                spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                states = []
                i = 0
                for row in spamreader:
                    if i < 5:
                        i=i+1
                        continue
                    row = [float(i) for i in row]
                    states.append(row)
            satellite["orbitpy_id"] = subdir
            satellite["states"] = states
            
        if "datametrics" in f:
            with open(directory+subdir+"/"+f,newline='') as csv_file:
                spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                visibilities = []
                i = 0
                for row in spamreader:
                    if i < 5:
                        i=i+1
                        continue
                    row[2] = "0.0"
                    row = [float(i) for i in row]
                    row.append(subdir)
                    visibilities.append(row)
            satellite["visibilities"] = visibilities
            #all_visibilities.extend(visibilities)


        if "plan_w_replan_interval.csv" in f:
            with open(directory+subdir+"/"+f,newline='') as csv_file:
                spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                observations_replan = []
                i = 0
                for row in spamreader:
                    if i < 1:
                        i=i+1
                        continue
                    row = [float(i) for i in row]
                    row.append(subdir)
                    observations_replan.append(row)
            print(len(unique(observations_replan)))
            satellite["observations_replan"] = observations_replan
            all_observations_replan.extend(observations_replan)

        if "plan.csv" == f:
            with open(directory+subdir+"/"+f,newline='') as csv_file:
                spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                observations = []
                i = 0
                for row in spamreader:
                    if i < 1:
                        i=i+1
                        continue
                    row = [float(i) for i in row]
                    row.append(subdir)
                    observations.append(row)
            satellite["observations"] = observations
            all_observations.extend(observations)

    if settings["preplanned_observations"] is not None:
        with open(settings["preplanned_observations"],newline='') as csv_file:
            csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
            observations = []
            i = 0
            for row in csvreader:
                if i < 1:
                    i=i+1
                    continue
                if int(row[0][8:]) == int(satellite["orbitpy_id"][3]):
                    obs = [int(float(row[3])),int(float(row[4])),float(row[1])*180/np.pi, float(row[2])*180/np.pi]
                    observations.append(obs)
        satellite["observations"] = observations
        

    if "orbitpy_id" in satellite:
        satellites.append(satellite)

all_visibilities = []
for satellite in satellites:
    vis_windows = []
    i = 0
    visibilities = satellite["visibilities"]
    while i < len(visibilities):
        continuous_visibilities = []
        visibility = visibilities[i]
        continuous_visibilities.append(visibility)
        start = visibility[0]*settings["step_size"]
        end = visibility[0]*settings["step_size"]
        while(i < len(visibilities)-1 and visibilities[i+1][0] == start):
            i += 1
        vis_done = False
        if i == len(visibilities)-1:
            break
        while not vis_done:
            vis_done = True
            num_steps = len(continuous_visibilities)
            while visibilities[i+1][0] == start+num_steps:
                if visibilities[i+1][1] == visibility[1]:
                    continuous_visibilities.append(visibilities[i+1])
                    end = visibilities[i+1][0]*settings["step_size"]
                    vis_done = False
                if i == len(visibilities)-2:
                    break
                else:
                    i += 1
            num_steps = len(continuous_visibilities)
            if i == len(visibilities)-1:
                break
        vis_window = [start,end,visibility[3],visibility[4],visibility[-1]]
        vis_windows.append(vis_window)
        for cont_vis in continuous_visibilities:
            visibilities.remove(cont_vis)
        i = 0
    print(len(vis_windows))
    all_visibilities.extend(vis_windows)
print(len(all_visibilities))
events = []
event_filename = './events/lakes/all_events.csv'
with open(event_filename,newline='') as csv_file:
    csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if i < 1:
            i=i+1
            continue
        events.append(row)

print("Actual observations")
compute_statistics(events,all_observations,"obs",10)
print("Actual observations, replan")
compute_statistics(events,all_observations_replan,"obs")
print("Potential observations (visibilities)")
all_visibilities.sort(key=lambda all_visibilities: all_visibilities[0])
print(len(all_visibilities))
event_chunks = list(chunks(events,10))
pool = multiprocessing.Pool()
input_list = []
for i in range(len(event_chunks)):
    input = {}
    input["events"] = event_chunks[i]
    input["observations"] = all_visibilities
    input_list.append(input)
output_list = pool.map(compute_statistics_pieces, input_list)
all_events_count_sum = 0
bloom_events_count_sum = 0
temp_events_count_sum = 0
level_events_count_sum = 0
all_events_reward_sum = 0
bloom_events_reward_sum = 0
temp_events_reward_sum = 0
level_events_reward_sum = 0
observed_locations_sum = 0
unique_observed_locations_sum = 0
for output in output_list:
    all_events_count_sum += output["all_events_count"]
    bloom_events_count_sum += output["bloom_events_count"]
    temp_events_count_sum += output["temp_events_count"]
    level_events_count_sum += output["level_events_count"]
    all_events_reward_sum += output["all_events_reward"]
    bloom_events_reward_sum += output["bloom_events_reward"]
    temp_events_reward_sum += output["temp_events_reward"]
    level_events_reward_sum += output["level_events_reward"]
    observed_locations_sum += output["num_observations"]
    unique_observed_locations_sum += output["num_unique_observations"]

print("All events co-observed: "+str(all_events_count_sum))
print("Bloom events co-observed: "+str(bloom_events_count_sum))
print("Temperature events co-observed: "+str(temp_events_count_sum))
print("Level events co-observed: "+str(level_events_count_sum))

print("All events co-observed reward: "+str(all_events_reward_sum))
print("Bloom events co-observed reward: "+str(bloom_events_reward_sum))
print("Temperature events co-observed reward: "+str(temp_events_reward_sum))
print("Level events co-observed reward: "+str(level_events_reward_sum))

print("Observed locations: "+str(observed_locations_sum))
print("Unique observed locations: "+str(unique_observed_locations_sum))