import numpy as np
import random


V = []
NQ = []

def check_maneuver_feasibility(curr_angle,obs_angle,curr_time,obs_end_time):
    """
    Checks to see if the specified angle change violates the maximum slew rate constraint.
    """
    moved = False
    # TODO add back FOV free visibility
    if(obs_end_time==curr_time):
        return False, False
    slew_rate = abs(obs_angle-curr_angle)/abs(obs_end_time-curr_time)
    max_slew_rate = 1.0 # deg / s
    #slewTorque = 4 * abs(np.deg2rad(new_angle)-np.deg2rad(curr_angle))*0.05 / pow(abs(new_time-curr_time),2)
    #maxTorque = 4e-3
    transition_end_time = abs(obs_angle-curr_angle)/max_slew_rate + curr_time
    moved = True
    return slew_rate < max_slew_rate, transition_end_time

def get_action_space(settings,state,obs_list):
    action_list = []
    i = 0
    while len(action_list) < settings["action_space_size"] and i < len(obs_list):
        feasible = False
        if obs_list[i]["start"] > state["time"]:
            feasible, transition_end_time = check_maneuver_feasibility(state["angle"],obs_list[i]["angle"],state["time"],obs_list[i]["end"])
        if feasible:
            action_list.append(obs_list[i])
        i = i+1
    return action_list

def transition_function(state,action):
    new_state = {
        "angle": action["angle"],
        "time": action["end"]
    }
    return new_state


def rollout(settings,state,action_space,obs_list,d):
    if d == 0:
        return 0
    if len(action_space) == 0:
        return 0
    else:
        selected_action = action_space[np.random.randint(len(action_space))]
        reward = selected_action["reward"]
        new_state = transition_function(state,selected_action)
        return (reward + rollout(settings,new_state,get_action_space(settings,new_state,obs_list),obs_list,(d-1)) * np.power(settings["gamma"],selected_action["start"]-state["time"]))

def simulate(settings,state,d,obs_list):
    if d == 0:
        return 0
    state_in_v = False
    for v in V:
        if state == v["state"]:
            state_in_v = True
    if not state_in_v:
        action_space = get_action_space(settings,state,obs_list)
        if action_space is None:
            return 0
        for action in action_space:
            state_action_pair = {
                "state": state,
                "action": action,
            }
            NQ_dict = {
                "sap": state_action_pair,
                "n_val": 1,
                "q_val": 0.0
            }
            NQ.append(NQ_dict)
            V.append(state_action_pair)
        return rollout(settings,state,action_space,obs_list,settings["solve_depth_init"])
    max = 0.0
    best_action = None
    n_sum = 0
    for nq in NQ:
        if nq["sap"]["state"] == state:
            n_sum = n_sum + nq["n_val"]
            q_val = nq["q_val"] + settings["c"]*np.sqrt(np.log10(n_sum)/nq["n_val"])
            if q_val > max:
                max = q_val
                best_action = nq["sap"]["action"]
    if best_action is None:
        return 0
    new_state = transition_function(state,best_action)
    r = best_action["reward"]
    q = r + simulate(settings,new_state,d-1,obs_list) * np.power(settings["gamma"],best_action["start"]-state["time"])
    best_sap = {
        "state": state,
        "action": best_action
    }
    for nq in NQ:
        if nq["sap"] == best_sap:
            nq["n_val"] += 1
            nq["q_val"] += (q-nq["q_val"]/nq["n_val"])
    return q


def monte_carlo_tree_search(obs_list):
    result_list = []
    initial_state = {
        "angle": 0,
        "time": 0
    }
    settings = {
        "n_max_sim": 50,
        "solve_depth_init": 50,
        "c": 3,
        "action_space_size": 4, 
        "gamma": 0.995
    }
    more_actions = True
    state = initial_state
    while more_actions:
        for n in range(settings["n_max_sim"]):
            simulate(settings,state,settings["solve_depth_init"],obs_list)
        max = 0
        best_action = None
        for nq in NQ:
            if nq["sap"]["state"] == state:
                value = nq["q_val"]
                if value > max:
                    max = value
                    best_action = nq["sap"]["action"]
        if best_action is None:
            break
        best_sap = {
            "state": state,
            "action": best_action
        }
        result_list.append(best_sap)
        state = transition_function(state,best_action)
        print(state["time"])
        more_actions = len(get_action_space(settings,state,obs_list)) != 0
    #print(V)
    #print(NQ)
    print(obs_list)
    planned_obs_list = []
    for result in result_list:
        planned_obs_list.append(result["action"])
    print(len(planned_obs_list))
    return planned_obs_list