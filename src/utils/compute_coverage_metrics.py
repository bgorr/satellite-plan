import csv
import datetime
import numpy as np
import os

def compute_percent_coverage_in_time(observations,points,time):
    obs_locations = []
    for obs in observations:
        if [obs[2],obs[3]] not in obs_locations and obs[0] < time:
            obs_locations.append([obs[2],obs[3]])
    return len(obs_locations)/len(points)

def compute_coverage_metrics(settings):
    directory = settings["directory"]+"orbit_data/"

    satellites = []
    all_initial_observations = []
    all_replan_observations = []
    all_visibilities = []


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

            if "plan" in f and not "replan" in f and settings["planner"] in f:
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
                all_initial_observations.extend(observations)

            if "replan" in f and settings["planner"] in f:
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
                all_replan_observations.extend(observations)

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
            start = visibility[0]
            end = visibility[0]
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
                        end = visibilities[i+1][0]
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
        all_visibilities.extend(vis_windows)

    events = []
    event_filename = settings["event_csvs"][0]
    with open(event_filename,newline='') as csv_file:
        csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        i = 0
        for row in csvreader:
            if i < 1:
                i=i+1
                continue
            events.append(row) # lat, lon, start, duration, severity
    
    grid_locations = []
    if not "point_grid" in settings:
        settings["point_grid"] = settings["directory"]+"orbit_data/grid0.csv"
    with open(settings["point_grid"],'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_locations.append([float(row[0]),float(row[1])])

    for i in range(864):
        time = i*100
        print("Time"+str(time))
        perc_cov = compute_percent_coverage_in_time(all_visibilities,grid_locations,time)
        print("Percent coverage: "+str(perc_cov))
        if perc_cov == 1.0:
            break
    return time

def main():
    cross_track_ffor = 15 # deg
    along_track_ffor = 15 # deg
    cross_track_ffov = 0 # deg
    along_track_ffov = 0 # deg
    agility = 1 # deg/s
    num_planes = 10
    num_sats_per_plane = 10
    settings = {
        "directory": "./missions/100_sats_prelim/",
        "step_size": 10,
        "duration": 1,
        "plot_interval": 5,
        "plot_duration": 2/24,
        "plot_location": ".",
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "uniform", # can be "event" or "static"
        "preplanned_observations": None,
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
        "planner": "dp",
        "process_obs_only": False
    }
    compute_coverage_metrics(settings)