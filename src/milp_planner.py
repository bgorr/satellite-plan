from pyscipopt import Model
import numpy as np
import csv
from utils.parse_SCIP_sols import *

def unique(lakes):
    lakes = np.asarray(lakes)
    return np.unique(lakes,axis=0)

def close_enough(lat0,lon0,lat1,lon1):
    if np.sqrt((lat0-lat1)**2+(lon0-lon1)**2) < 0.01:
        return True
    else:
        return False

def milp_planner(satellites,settings):
    prefix = settings["experiment_settings"]["name"]
    access_file = "./"+prefix+"_milp_accesses.csv"
    input_filename = "./"+prefix+"_milp_input.zpl"
    output_filename = "./"+prefix+"_milp_output.sol"
    s = 0
    times = set()
    locations = set()
    for satellite in satellites:
        for obs in satellite["obs_list"]:
            locations.add((obs["location"]["lat"]/180*np.pi,obs["location"]["lon"]/180*np.pi))
            times.add(obs["end"]*settings["step_size"])
    times = list(times)
    locations = list(locations)
    rows = []
    rows.append(["sat","lat","lon","rise_time","set_time","incidence_angle","reward","s","p","i","t","obs"])
    for satellite in satellites:
        for obs in satellite["obs_list"]:
            obs2 = 1
            boundaries = np.linspace(0,30*np.pi/180,6)
            p = np.searchsorted(boundaries, abs(obs["angle"])/180*np.pi)+7
            i = locations.index((obs["location"]["lat"]/180*np.pi,obs["location"]["lon"]/180*np.pi))+1
            t = times.index(obs["end"]*settings["step_size"])+1
            row = [satellite["orbitpy_id"], obs["location"]["lat"]/180*np.pi, obs["location"]["lon"]/180*np.pi, obs["start"], obs["end"], abs(obs["angle"])/180*np.pi, obs["reward"], s,p,i,t,obs2] # sat,lat,lon,rise_time,set_time,incidence_angle,reward,s,p,i,t,obs
            rows.append(row)
        s += 1
    
    with open(access_file,'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            csvwriter.writerow(row)
    model = Model()
    base_filename = "./src/utils/MILP_slew_base.zpl"
    file = open(base_filename,"r")
    newfile = open(input_filename, "w")

    for line in file:
        if "read" in line:
            tokens = line.split(" ")
            tokens[1] = "\""+access_file+"\""
            line = " ".join(tokens)
            newfile.write(line)
        elif "Smax" in line and not "set" in line:
            tokens = line.split(" ")
            tokens[3] = str(9)+";"
            line = " ".join(tokens)
            newfile.write(line)
        elif "Pmax" in line and not "set" in line:
            tokens = line.split(" ")
            tokens[3] = str(13)+";"
            line = " ".join(tokens)
            newfile.write(line)
        elif "Tmax" in line and not "set" in line:
            tokens = line.split(" ")
            tokens[3] = str(len(times))+";"
            line = " ".join(tokens)
            newfile.write(line)
        elif "Imax" in line and not "set" in line:
            tokens = line.split(" ")
            tokens[3] = str(len(locations))+";"
            line = " ".join(tokens)
            newfile.write(line)
        elif "max_torque := " in line:
            tokens = line.split(" ")
            tokens[3] = str(settings["experiment_settings"]["agility"])+";"
            line = " ".join(tokens)
            newfile.write(line)
        else:
            newfile.write(line)
    file.close()
    newfile.close()
    model.readProblem(input_filename)
    model.optimize()
    sol = model.getBestSol()
    model.writeBestSol(output_filename)
    obs = write_plan(output_filename,access_file,"./"+prefix+"_milp_output_plan.csv")
    return obs

def milp_planner_interval(planner_input_list):
    settings = planner_input_list[0]["settings"]
    prefix = settings["experiment_settings"]["name"]
    access_file = "./"+prefix+"_milp_accesses.csv"
    input_filename = "./"+prefix+"_milp_input.zpl"
    output_filename = "./"+prefix+"_milp_output.sol"
    s = 0
    times = set()
    locations = set()
    satellite_list = []

    for planner_input in planner_input_list:
        satellite_list.append(planner_input["orbitpy_id"])
        events = planner_input["events"]
        settings = planner_input["settings"]
        plan_end = planner_input["plan_end"]
        sharing_end = planner_input["sharing_end"]
        for obs in planner_input["obs_list"]:
            locations.add((obs["location"]["lat"]/180*np.pi,obs["location"]["lon"]/180*np.pi))
            times.add(obs["end"]*settings["step_size"])
    times = list(times)
    locations = list(locations)
    rows = []
    rows.append(["sat","lat","lon","rise_time","set_time","incidence_angle","reward","s","p","i","t","obs"])
    for planner_input in planner_input_list:
        for obs in planner_input["obs_list"]:
            obs2 = 1
            boundaries = np.linspace(0,30*np.pi/180,6)
            p = np.searchsorted(boundaries, abs(obs["angle"])/180*np.pi)+7
            i = locations.index((obs["location"]["lat"]/180*np.pi,obs["location"]["lon"]/180*np.pi))+1
            t = times.index(obs["end"]*settings["step_size"])+1
            row = [planner_input["orbitpy_id"], obs["location"]["lat"]/180*np.pi, obs["location"]["lon"]/180*np.pi, obs["start"], obs["end"], abs(obs["angle"])/180*np.pi, obs["reward"], s,p,i,t,obs2] # sat,lat,lon,rise_time,set_time,incidence_angle,reward,s,p,i,t,obs
            rows.append(row)
        s += 1
    
    with open(access_file,'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            csvwriter.writerow(row)
    model = Model()
    base_filename = "./src/utils/MILP_slew_base.zpl"
    file = open(base_filename,"r")
    newfile = open(input_filename, "w")

    for line in file:
        if "read" in line:
            tokens = line.split(" ")
            tokens[1] = "\""+access_file+"\""
            line = " ".join(tokens)
            newfile.write(line)
        elif "Smax" in line and not "set" in line:
            tokens = line.split(" ")
            tokens[3] = str(9)+";"
            line = " ".join(tokens)
            newfile.write(line)
        elif "Pmax" in line and not "set" in line:
            tokens = line.split(" ")
            tokens[3] = str(13)+";"
            line = " ".join(tokens)
            newfile.write(line)
        elif "Tmax" in line and not "set" in line:
            tokens = line.split(" ")
            tokens[3] = str(len(times))+";"
            line = " ".join(tokens)
            newfile.write(line)
        elif "Imax" in line and not "set" in line:
            tokens = line.split(" ")
            tokens[3] = str(len(locations))+";"
            line = " ".join(tokens)
            newfile.write(line)
        elif "max_torque := " in line:
            tokens = line.split(" ")
            tokens[3] = str(settings["experiment_settings"]["agility"])+";"
            line = " ".join(tokens)
            newfile.write(line)
        else:
            newfile.write(line)
    file.close()
    newfile.close()
    model.readProblem(input_filename)
    model.optimize()
    sol = model.getBestSol()
    model.writeBestSol(output_filename)
    planned_obs_list = write_plan(output_filename,access_file,"./"+prefix+"_milp_output_plan.csv")
    planner_outputs = []
    for satellite in satellite_list:
        prelim_plan = []
        for obs in planned_obs_list:
            if obs[0] == satellite:
                prelim_plan.append(obs[1:])
        plan = []
        updated_rewards = []
        for next_obs in prelim_plan:
            next_obs_dict = {
                "location": {
                    "lat": float(next_obs[0])*180/np.pi,
                    "lon": float(next_obs[1])*180/np.pi
                },
                "start": float(next_obs[2]),
                "end": float(next_obs[3]),
                "soonest": float(next_obs[3]),
                "angle": float(next_obs[4])*180/np.pi,
                "reward": float(next_obs[5])
            }
            next_obs = next_obs_dict
            plan.append(next_obs)
            curr_time = next_obs["end"]
            not_in_event = True
            for event in events:
                if close_enough(next_obs["location"]["lat"],next_obs["location"]["lon"],event["location"]["lat"],event["location"]["lon"]):
                    if (event["start"] <= next_obs["start"] <= event["end"]) or (event["start"] <= next_obs["end"] <= event["end"]) and next_obs["end"] < sharing_end:
                        updated_reward = { 
                            "reward": event["severity"]*settings["reward"],
                            "location": next_obs["location"],
                            "last_updated": curr_time,
                            "orbitpy_id": satellite
                        }
                        updated_rewards.append(updated_reward)
                        not_in_event = False
            if not_in_event and next_obs["end"] < sharing_end:
                updated_reward = {
                    "reward": 0.0,
                    "location": next_obs["location"],
                    "last_updated": curr_time,
                    "orbitpy_id": satellite
                }
                updated_rewards.append(updated_reward)
            if curr_time > plan_end:
                break
        planner_output = {
            "plan": plan,
            "end_time": plan_end,
            "updated_rewards": updated_rewards
        }
        planner_outputs.append(planner_output)
    return planner_outputs

def main():
    milp_planner("./src/utils/plan_milp.zpl","/home/ben/repos/satplan/src/utils/accesses_2h_5sat_processed.csv","./src/utils/plan_milp.sol")

if __name__ == "__main__":
    main()