import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator

def get_nonzero_observations(input_str):
    input_str = input_str.replace('[','')
    input_str = input_str.replace(']','')
    input_str = input_str.replace('\n','')
    input_array = input_str.split(',')
    input_farray = np.asfarray(input_array)
    return input_farray[input_farray!=0]

plot_dir = "./plots/plot_uniform_reobs/"
if not os.path.exists(plot_dir):
    os.mkdir("./plots/plot_uniform_reobs/")

rows = {}
with open("./results/uniform_reobs_study.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        rows[row[0]] = row

fig, ax1 = plt.subplots()
plt.rcParams.update({'font.size': 12})

N = 4
ind = np.arange(N)+1
width = 0.125
distribution = [1,2,3,4]
row = rows["uniform_reobs_study"]
initial_observations = get_nonzero_observations(row[23])
lin_inc_observations = get_nonzero_observations(row[33])

row = rows["uniform_reobs_study_0"]
print(row[33])
lin_dec_observations = get_nonzero_observations(row[33])

row = rows["uniform_reobs_study_1"]
dec_dec_observations = get_nonzero_observations(row[33])

row = rows["uniform_reobs_study_2"]
dec_inc_observations = get_nonzero_observations(row[33])

row = rows["uniform_reobs_study_3"]
imm_dec_observations = get_nonzero_observations(row[33])

row = rows["uniform_reobs_study_4"]
no_change_observations = get_nonzero_observations(row[33])

all_observations = []
labels = []
all_observations.extend(initial_observations)
labels.extend(['Initial']*len(initial_observations))
all_observations.extend(lin_inc_observations)
labels.extend(['Linear Increase']*len(lin_inc_observations))
all_observations.extend(lin_dec_observations)
labels.extend(['Linear Decrease']*len(lin_dec_observations))
all_observations.extend(dec_dec_observations)
labels.extend(['Decaying Decrease']*len(dec_dec_observations))
all_observations.extend(dec_inc_observations)
labels.extend(['Decaying Increase']*len(dec_inc_observations))
all_observations.extend(imm_dec_observations)
labels.extend(['Immediate Decrease']*len(imm_dec_observations))
all_observations.extend(no_change_observations)
labels.extend(['No Change']*len(no_change_observations))
all_data = {'Reobserving Strategy': labels,'Number of re-observations': all_observations}
all_df = pd.DataFrame(data=all_data)

number_of_reobs = []
initial_reobs = []
lin_inc_reobs = []
lin_dec_reobs = []
dec_dec_reobs = []
dec_inc_reobs = []
imm_dec_reobs = []
no_change_reobs = []
for i in range(int(np.max([np.max(lin_inc_observations),np.max(initial_observations),np.max(lin_dec_observations),np.max(dec_inc_observations),np.max(dec_dec_observations),np.max(imm_dec_observations),np.max(no_change_observations)]))+1):
    number_of_reobs.append(i)
    init_obs_count = 0
    for init_obs in initial_observations:
        if init_obs == i:
            init_obs_count += 1
    initial_reobs.append(init_obs_count)
    lin_inc_obs_count = 0
    for lin_inc_obs in lin_inc_observations:
        if lin_inc_obs == i:
            lin_inc_obs_count += 1
    lin_inc_reobs.append(lin_inc_obs_count)
    dec_inc_obs_count = 0
    for dec_inc_obs in dec_inc_observations:
        if dec_inc_obs == i:
            dec_inc_obs_count += 1
    dec_inc_reobs.append(dec_inc_obs_count)
    dec_dec_obs_count = 0
    for dec_dec_obs in dec_dec_observations:
        if dec_dec_obs == i:
            dec_dec_obs_count += 1
    dec_dec_reobs.append(dec_dec_obs_count)
    lin_dec_obs_count = 0
    for lin_dec_obs in lin_dec_observations:
        if lin_dec_obs == i:
            lin_dec_obs_count += 1
    lin_dec_reobs.append(lin_dec_obs_count)
    imm_dec_obs_count = 0
    for imm_dec_obs in imm_dec_observations:
        if imm_dec_obs == i:
            imm_dec_obs_count += 1
    imm_dec_reobs.append(imm_dec_obs_count)
    no_change_obs_count = 0
    for no_change_obs in no_change_observations:
        if no_change_obs == i:
            no_change_obs_count += 1
    no_change_reobs.append(no_change_obs_count)

plt.plot(number_of_reobs,initial_reobs,color="black",linestyle=":",label="Non-reactive")
plt.plot(number_of_reobs,lin_inc_reobs,color="red",linestyle='--',label="Linear increase")
plt.plot(number_of_reobs,lin_dec_reobs,color="green",linestyle='-.',label="Linear decrease")
plt.plot(number_of_reobs,dec_dec_reobs,color="blue",linestyle='-',label="Decaying decrease")
plt.plot(number_of_reobs,dec_inc_reobs,color="yellow",linestyle=':',label="Decaying increase")
plt.plot(number_of_reobs,imm_dec_reobs,color="orange",linestyle='--',label="Immediate decrease")
plt.plot(number_of_reobs,no_change_reobs,color="purple",linestyle='-.',label="No change")
plt.xlabel("Number of re-observations")
plt.ylabel("Quantity")
plt.legend()

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig(plot_dir+"/uniform_reobs_linear.png",dpi=300, bbox_inches="tight")

plt.close()