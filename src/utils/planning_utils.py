import numpy as np

def get_action_space(curr_time,curr_angle,obs_list,last_obs,settings):
    feasible_actions = []
    for obs in obs_list:
        if last_obs is not None and obs["location"]["lat"] == last_obs["location"]["lat"]:
            continue
        if obs["start"] > curr_time:
            feasible, transition_end_time = check_maneuver_feasibility(curr_angle,np.min(obs["angles"]),curr_time,obs["end"],settings)
            if transition_end_time < obs["start"]:
                obs["soonest"] = obs["start"]
            else:
                obs["soonest"] = transition_end_time
            if feasible:
                feasible_actions.append(obs)
        if len(feasible_actions) > 10: # THIS IS NOT A GOOD IDEA BUT SHOULD HELP RUNTIME TODO
            break
    return feasible_actions

def get_action_space_kg(curr_time,curr_angle,obs_list,last_obs,settings,):
    feasible_actions = []
    for obs in obs_list:
        if last_obs is not None and obs["location"]["lat"] == last_obs["location"]["lat"]:
            continue
        if obs["start"] > curr_time:
            feasible, transition_end_time = check_maneuver_feasibility(curr_angle,np.min(obs["angles"]),curr_time,obs["end"],settings)
            if transition_end_time < obs["start"]:
                obs["soonest"] = obs["start"]
            else:
                obs["soonest"] = transition_end_time
            if feasible:
                feasible_actions.append(obs)
        if len(feasible_actions) > 10: # THIS IS NOT A GOOD IDEA BUT SHOULD HELP RUNTIME TODO
            break
    return feasible_actions


def check_maneuver_feasibility(curr_angle,obs_angle,curr_time,obs_end_time,settings):
    """
    Checks to see if the specified angle change violates the maximum slew rate constraint.
    """
    # TODO add back FOV free visibility
    if(obs_end_time==curr_time):
        return False, False
    
    if settings["agility"]["slew_constraint"] == "torque":
        obs_end_time = obs_end_time*settings["time"]["step_size"]
        curr_time = curr_time*settings["time"]["step_size"]
        obs_angle = 5 * round(obs_angle/5)
        curr_angle = 5 * round(curr_angle/5)
        inertia = settings["agility"]["inertia"]
        slew_torque = 4 * abs(np.deg2rad(obs_angle)-np.deg2rad(curr_angle))*inertia / pow(abs(obs_end_time-curr_time),2)
        max_torque = settings["agility"]["max_torque"]
        transition_end_time = (np.sqrt(4 * abs(np.deg2rad(obs_angle)-np.deg2rad(curr_angle))*inertia / max_torque) + curr_time)/settings["time"]["step_size"]
        
        return slew_torque < max_torque, transition_end_time
    elif settings["agility"]["slew_constraint"] == "rate":
        slew_rate = abs(obs_angle-curr_angle)/abs(obs_end_time-curr_time)/settings["time"]["step_size"]
        max_slew_rate = settings["agility"]["max_slew_rate"] # deg / s
        transition_end_time = abs(obs_angle-curr_angle)/(max_slew_rate*settings["time"]["step_size"]) + curr_time
        # if curr_angle == -32.91:
        #     print(slew_rate)
        #     print(obs_angle)
        #     print(curr_angle)
        #     print(max_slew_rate)
        #     print(obs_end_time)
        #     print(curr_time)
        #     print(settings["time"]["step_size"])
        # if slew_rate < max_slew_rate and transition_end_time > obs_end_time:
        #     print(transition_end_time)
        #     print(obs_end_time)
        feasible = (slew_rate < max_slew_rate)# and curr_time < obs_end_time and transition_end_time < obs_end_time
        return feasible, transition_end_time
    else:
        print("Invalid slewing constraint provided")
        return False, False

def close_enough(lat0,lon0,lat1,lon1):
    if np.sqrt((lat0-lat1)**2+(lon0-lon1)**2) <= 0.0001:
        return True
    else:
        return False