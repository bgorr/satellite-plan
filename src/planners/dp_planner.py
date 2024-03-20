import numpy as np
import random
import sys
from tqdm import tqdm
from src.utils.planning_utils import check_maneuver_feasibility

#sys.path.append('/home/ben/repos/UKGE/UKGE')
#sys.path.append('/home/ben/repos/UKGE/UKGE/src')
#import testers

V = []
NQ = []

def filter_obs_list_kg(obs_list,sat_id):
    print("Filtering obs list with kg")
    instruments = ['AIRS','OLI','OLCI','DORIS-NG','DESIS','EMIT','NIRST','POSEIDON-3C Altimeter','PSA']
    instrument = instruments[int(sat_id[3:])]
    filtered_obs_list = []
    for obs in tqdm(obs_list):
        parameter = obs["parameter"]
        if query_KG(instrument,parameter):
            filtered_obs_list.append(obs)
    return filtered_obs_list

def filter_obs_list_ukge(obs_list,sat_id,settings):
    print("Filtering obs list with ukge")
    instruments = ['AIRS','OLI','OLCI','DORIS-NG','DESIS','EMIT','NIRST','POSEIDON-3C Altimeter','PSA']
    instrument = instruments[int(sat_id[3:])]
    filtered_obs_list = []
    for obs in tqdm(obs_list):
        parameter = obs["parameter"]
        print(settings["ukge_threshold"])
        if query_UKGE(instrument,parameter) > settings["ukge_threshold"]:
            filtered_obs_list.append(obs)
    return filtered_obs_list

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
        

def close_enough(lat0,lon0,lat1,lon1):
    if np.sqrt((lat0-lat1)**2+(lon0-lon1)**2) < 0.01:
        return True
    else:
        return False

def propagate_weights(obs_list,settings):
    rewards = np.zeros(shape=(len(obs_list)))
    node_indices = [None]*len(obs_list)
    for i in range(len(obs_list)):
        rewards[i] = obs_list[i]["reward"]
        node_indices[i] = None
    for i in range(len(obs_list)):
        for k in range(len(obs_list)):
            feasible, transition_end_time = check_maneuver_feasibility(obs_list[i]["angle"],obs_list[k]["angle"],obs_list[i]["start"],obs_list[k]["end"],settings)
            if transition_end_time < obs_list[i]["start"]:
                obs_list[i]["soonest"] = obs_list[i]["start"]
            else:
                obs_list[i]["soonest"] = transition_end_time
            if feasible and ((rewards[i]+obs_list[k]["reward"]) > rewards[k]) and obs_list[i]["soonest"] < obs_list[k]["start"]:
                rewards[k] = rewards[i] + obs_list[k]["reward"]
                node_indices[k] = i
    return rewards, node_indices

def extract_path(obs_list,rewards,node_indices):
    reverse_plan = []
    if not rewards.any():
        return []
    node_index = np.argmax(rewards)
    while node_index is not None:
        reverse_plan.append(obs_list[node_index])
        node_index = node_indices[node_index]
    plan = reversed(reverse_plan)
    return plan

def graph_search(obs_list,settings):
    rewards, node_indices = propagate_weights(obs_list,settings)
    plan = extract_path(obs_list,rewards,node_indices)
    return list(plan)

def graph_search_kg(obs_list,sat_id,settings):
    filtered_obs_list = filter_obs_list_kg(obs_list,sat_id)
    rewards, node_indices = propagate_weights(filtered_obs_list,settings)
    plan = extract_path(filtered_obs_list,rewards,node_indices)
    return list(plan)

def graph_search_ukge(obs_list,sat_id,settings):
    filtered_obs_list = filter_obs_list_ukge(obs_list,sat_id,settings)
    rewards, node_indices = propagate_weights(filtered_obs_list,settings)
    plan = extract_path(filtered_obs_list,rewards,node_indices)
    return list(plan)

def graph_search_events(planner_inputs):
    events = planner_inputs["events"]
    obs_list = planner_inputs["obs_list"]
    plan_start = planner_inputs["plan_start"] # TODO chop the search
    plan_end = planner_inputs["plan_end"]
    settings = planner_inputs["settings"]
    rewards, node_indices = propagate_weights(obs_list,settings)
    prelim_plan = list(extract_path(obs_list,rewards,node_indices))
    if len(prelim_plan) == 0:
        planner_outputs = {
            "plan": [],
            "end_time": plan_end,
            "updated_reward": None
        }
        return planner_outputs
    plan = []
    for next_obs in prelim_plan:
        plan.append(next_obs)
        curr_time = next_obs["soonest"]
        for event in events:
            if close_enough(next_obs["location"]["lat"],next_obs["location"]["lon"],event["location"]["lat"],event["location"]["lon"]):
                if (event["start"] <= next_obs["start"] <= event["end"]) or (event["start"] <= next_obs["end"] <= event["end"]):
                    updated_reward = { 
                        "reward": event["severity"]*settings["rewards"]["reward"],
                        "location": next_obs["location"],
                        "last_updated": curr_time 
                    }
                    planner_outputs = {
                        "plan": plan,
                        "end_time": curr_time,
                        "updated_reward": updated_reward
                    }
                    return planner_outputs
    planner_outputs = {
        "plan": plan,
        "end_time": plan_end,
        "updated_reward": None
    }
    return planner_outputs

def graph_search_events_interval(planner_inputs):
    events = planner_inputs["events"]
    obs_list = planner_inputs["obs_list"]
    plan_start = planner_inputs["plan_start"]
    plan_end = planner_inputs["plan_end"]
    sharing_end = planner_inputs["sharing_end"]
    settings = planner_inputs["settings"]
    orbitpy_id = planner_inputs["orbitpy_id"]
    start_angle = planner_inputs["start_angle"]
    filtered_obs_list = []
    for obs in obs_list:
        feasible, _ = check_maneuver_feasibility(start_angle,obs["angle"],plan_start,obs["end"],settings)
        if feasible:
            filtered_obs_list.append(obs)
    obs_list = filtered_obs_list
    rewards, node_indices = propagate_weights(obs_list,settings)
    prelim_plan = list(extract_path(obs_list,rewards,node_indices))
    updated_rewards = []
    if len(prelim_plan) == 0:
        planner_outputs = {
            "plan": [],
            "end_time": plan_end,
            "updated_rewards": updated_rewards
        }
        return planner_outputs
    plan = []
    for next_obs in prelim_plan:
        plan.append(next_obs)
        curr_time = next_obs["end"]
        not_in_event = True
        for event in events:
            if close_enough(next_obs["location"]["lat"],next_obs["location"]["lon"],event["location"]["lat"],event["location"]["lon"]):
                if (event["start"] <= next_obs["start"] <= event["end"]) or (event["start"] <= next_obs["end"] <= event["end"]) and next_obs["end"] < sharing_end:
                    updated_reward = { 
                        "reward": event["severity"]*settings["rewards"]["reward"],
                        "location": next_obs["location"],
                        "last_updated": curr_time,
                        "orbitpy_id": orbitpy_id
                    }
                    updated_rewards.append(updated_reward)
                    not_in_event = False
        if not_in_event and next_obs["end"] < sharing_end:
            updated_reward = {
                "reward": 0.0,
                "location": next_obs["location"],
                "last_updated": curr_time,
                "orbitpy_id": orbitpy_id
            }
            updated_rewards.append(updated_reward)
        if curr_time > plan_end:
            break
                    
    planner_outputs = {
        "plan": plan,
        "end_time": plan_end,
        "updated_rewards": updated_rewards
    }
    return planner_outputs