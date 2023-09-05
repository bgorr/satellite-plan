import numpy as np
import datetime
import csv
import os

def close_enough(lat0,lon0,lat1,lon1):
    if np.sqrt((lat0-lat1)**2+(lon0-lon1)**2) < 0.01:
        return True
    else:
        return False

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
            
        if "access" in f:
            with open(directory+subdir+"/"+f,newline='') as csv_file:
                spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                visibilities = []
                i = 0
                for row in spamreader:
                    if i < 5:
                        i=i+1
                        continue
                    row = [float(i) for i in row]
                    visibilities.append(row)
            satellite["visibilities"] = visibilities

        if "plan" in f:
            with open(directory+subdir+"/"+f,newline='') as csv_file:
                spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                observations = []
                i = 0
                for row in spamreader:
                    if i < 5:
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

all_events_count = 0
bloom_events_count = 0
temp_events_count = 0
level_events_count = 0

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
    #print(event_type)

    for obs in all_observations:
        if close_enough(obs[2],obs[3],float(event[0]),float(event[1])):
            if float(event[2]) < obs[0] < (float(event[2])+float(event[3])):
                if satellite_name_dict[obs[4]] in measurements:
                    measurements.remove(satellite_name_dict[obs[4]])

    if len(measurements) == 0:
        print("whoa a co-obs")
        if event_type == "bloom":
            bloom_events_count += 1
            all_events_count += 1
        if event_type == "temp":
            bloom_events_count += 1
            all_events_count += 1
        if event_type == "level":
            bloom_events_count += 1
            all_events_count += 1    
