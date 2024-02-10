import numpy as np
import datetime
import csv
import os
import multiprocessing
import sys
from functools import partial
from tqdm import tqdm

sys.path.append('/home/ben/repos/UKGE/UKGE')
sys.path.append('/home/ben/repos/UKGE/UKGE/src')
import testers

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

def compute_max_revisit_time(start,end,observations,settings):
    # only computing based on starts so that I don't have to do start/stop tracking
    # TODO stop being lazy
    start_list = []
    start_list.append(start)
    start_list.append(end)
    for obs in observations:
        start_list.append(obs[0]*settings["time"]["step_size"])
    start_list = np.asarray(start_list)
    start_list = np.sort(start_list)
    gaps = []
    for i in range(len(start_list)-1):
        gaps.append(start_list[i+1]-start_list[i])
    gaps = np.asarray(gaps)
    return np.max(gaps)

def compute_avg_revisit_time(start,end,observations,settings):
    # only computing based on starts so that I don't have to do start/stop tracking
    # TODO stop being lazy
    start_list = []
    start_list.append(start)
    start_list.append(end)
    for obs in observations:
        start_list.append(obs[0]*settings["time"]["step_size"])
    start_list = np.asarray(start_list)
    start_list = np.sort(start_list)
    gaps = []
    for i in range(len(start_list)-1):
        gaps.append(start_list[i+1]-start_list[i])
    gaps = np.asarray(gaps)
    return np.average(gaps)

def query_KG(instrument, parameter):
    in_kg = False
    with open("/home/ben/repos/UKGE/KG raw data/prob_kg.csv", "r") as kg_file:
        for line in kg_file.readlines():
            line.rstrip()
            tokens = line.split(",")
            if instrument == tokens[0] and "OBSERVES" == tokens[1] and parameter == tokens[2]:
                in_kg = True
                break
    return in_kg

def query_UKGE(instrument, parameter):
    model_dir = '/home/ben/repos/UKGE/UKGE/trained_models/3D_CHESS/rect_1030/'
    data_filename = 'data.bin'
    model_filename = 'model.bin'
    validator = testers.UKGE_rect_Tester()
    relation_name = 'OBSERVES'
    instrument_id = None
    parameter_id = None
    relation_id = None
    with open("/home/ben/repos/UKGE/KG processed data/en2id.txt", "r") as entity_file:
        for line in entity_file.readlines():
            line.rstrip()
            tokens = line.split("\t")
            if instrument == tokens[0]:
                instrument_id = tokens[1].rstrip()
            if parameter == tokens[0]:
                parameter_id = tokens[1].rstrip()
    with open("/home/ben/repos/UKGE/KG processed data/rel2id.txt", "r") as relation_file:
        for line in relation_file:
            line.rstrip()
            tokens = line.split("\t")
            if relation_name == tokens[0]:
                relation_id = tokens[1].rstrip()
    if instrument_id is None or parameter_id is None or relation_id is None:
        print(instrument_id)
        print(parameter_id)
        print(relation_id)
        print("invalid query")
        exit()
    new_query_filename = 'new_query.tsv'
    with open(new_query_filename, "w") as query_file:
        query_file.write(instrument_id + "\t" + relation_id + "\t" + parameter_id + "\t" + "1.00000" + "\n")
        query_file.write(instrument_id + "\t" + relation_id + "\t" + parameter_id + "\t" + "1.00000" + "\n")

    validator.build_by_file(new_query_filename, model_dir, model_filename, data_filename)
    scores = validator.get_val()
    return scores[0]

def compute_statistics_pieces(observations,ground_truth,threshold):
    #observations = input["observations"]
    instruments = ['AIRS','OLI','OLCI','DORIS-NG','DESIS','EMIT','NIRST','POSEIDON-3C Altimeter','PSA']
    filtered_obs_list = []
    for obs in tqdm(observations):
        sat_id = int(obs[6][3:])
        instrument = instruments[sat_id]
        parameter = obs[5]
        if ground_truth == "det_kg":
            if query_KG(instrument,parameter):
                filtered_obs_list.append(obs)
        elif ground_truth == "ukge":
            if query_UKGE(instrument,parameter) > threshold:
                filtered_obs_list.append(obs)
    return len(filtered_obs_list)

def compute_statistics(events,obs,grid_locations,settings):
    num_obs_det = compute_statistics_pieces(obs,"det_kg",0.0)
    num_obs_025 = compute_statistics_pieces(obs,"ukge",0.25)
    num_obs_05 = compute_statistics_pieces(obs,"ukge",0.5)
    num_obs_075 = compute_statistics_pieces(obs,"ukge",0.75)

    kg_statistics = {
        "num_obs_det": num_obs_det,
        "wasted_eff_det": len(obs)-num_obs_det,
        "num_obs_0.25": num_obs_025,
        "wasted_eff_0.25": len(obs)-num_obs_025,
        "num_obs_0.5": num_obs_05,
        "wasted_eff_0.5": len(obs)-num_obs_05,
        "num_obs_0.75": num_obs_075,
        "wasted_eff_0.75": len(obs)-num_obs_075,
    }
    return kg_statistics

def compute_experiment_statistics(settings):
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
        if ".csv" in subdir:
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
                        obs_row = [float(i) for i in row[0:-2]]
                        obs_row.append(row[-1])
                        obs_row.append(subdir)
                        observations.append(obs_row)
                    unique_observations = []
                    obs_end_times = []
                    for obs in observations:
                        if obs[1] not in obs_end_times:
                            obs_end_times.append(obs[1])
                            unique_observations.append(obs)
                    observations = unique_observations
                all_initial_observations.extend(observations)

            if "replan" in f and settings["planner"] in f and "het" not in f:
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
                    unique_observations = []
                    obs_end_times = []
                    for obs in observations:
                        if obs[1] not in obs_end_times:
                            obs_end_times.append(obs[1])
                            unique_observations.append(obs)
                    observations = unique_observations
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
            vis_window = [start,end,visibility[3],visibility[4],0,0,visibility[-1]] # no reward or angle associated with visibilities
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
    with open(settings["point_grid"],'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_locations.append([float(row[0]),float(row[1])])

    print("Initial event observations")
    kg_results = compute_statistics(events,all_initial_observations,grid_locations,settings)
    # print("Replan event observations")
    # replan_results = compute_statistics(events,all_replan_observations,grid_locations,settings)
    # print("Potential observations (visibilities)")
    # vis_results = compute_statistics(events,all_visibilities,grid_locations,settings)
    print("num_visibilities: "+str(len(all_visibilities)))
    overall_results = kg_results
    overall_results["num_vis"] = len(all_visibilities)
    return overall_results

def main():
    name = "compute_exp_stats"
    settings = {
        "name": name,
        "instrument": {
            "ffor": 30,
            "ffov": 0
        },
        "agility": {
            "slew_constraint": "rate",
            "max_slew_rate": 0.1
        },
        "orbit": {
            "altitude": 705, # km
            "inclination": 98.4, # deg
            "eccentricity": 0.0001,
            "argper": 0, # deg
        },
        "constellation": {
            "num_sats_per_plane": 6,
            "num_planes": 6,
            "phasing_parameter": 1
        },
        "events": {
            "event_duration": 3600*6,
            "event_frequency": 0.01/3600,
            "event_density": 2,
            "event_clustering": 4
        },
        "time": {
            "step_size": 10, # seconds
            "duration": 1, # days
            "initial_datetime": datetime.datetime(2020,1,1,0,0,0)
        },
        "rewards": {
            "reward": 10,
            "reward_increment": 0.1,
        },
        "planner": "milp",
        "num_meas_types": 3,
        "sharing_horizon": 1000,
        "planning_horizon": 1000,
        "directory": "./missions/"+name+"/",
        "grid_type": "event", # can be "event" or "static"
        "point_grid": "./coverage_grids/"+name+"/event_locations.csv",
        "preplanned_observations": None,
        "event_csvs": ["./events/"+name+"/events.csv"],
        "process_obs_only": False,
    }
    overall_results = compute_experiment_statistics(settings)
    print(overall_results)

if __name__ == "__main__":
    main()