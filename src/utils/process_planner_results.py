import csv
import numpy as np

def close_enough(lat0,lon0,lat1,lon1):
    if np.sqrt((lat0-lat1)**2+(lon0-lon1)**2) < (0.001*np.pi/180):
        return True
    else:
        return False
    
def get_reward(lat,lon,angle,rewards):
    reward = 0
    for rew in rewards:
        if close_enough(lat,lon,float(rew[0]),float(rew[1])):
            reward += float(rew[2])
    reward *= (1-angle)
    return reward

rewards = []
# file is formatted [lat,lon,reward]
with open('./rewards.csv',newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        rewards.append(row)

# assume that observations are sorted
# file is formatted [lat,lon,start_time,end_time,incidence_angle], all in radians
observations = []
with open('./observations.csv',newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        observations.append(row)

already_observed_locations = []
reobservations_penalized = True
curr_time = 0
total_reward = 0
for obs in observations:
    if float(obs[2]) < curr_time:
        print("Observations are not sorted.")
    curr_time = float(obs[2])
    already_observed = False
    if len(already_observed_locations) > 0:
        for already_obs in already_observed_locations:
            if close_enough(float(obs[0]),float(obs[1]),already_obs[0],already_obs[1]):
                already_observed = True
    already_observed_locations.append((float(obs[0]),float(obs[1])))
    if already_observed and reobservations_penalized:
        reward = 0
    else:
        reward = get_reward(float(obs[0]),float(obs[1]),float(obs[4]),rewards)
    total_reward += reward

print("Total reward: "+str(total_reward))