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


rows = {}
plot_dir = "./plots/fire_plots/"
if not os.path.exists(plot_dir):
    os.mkdir("./plots/fire_plots/")

with open("./studies/fire_constellation_study_decdec.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        rows[row[0]] = row

fig, ax1 = plt.subplots()
plt.rcParams.update({'font.size': 12})

for row_key in rows.keys():
    fig, ax1 = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    row = rows[row_key]
    initial_observations = get_nonzero_observations(row[23])
    replan_observations = get_nonzero_observations(row[38])

    initial_observations = initial_observations - 1
    replan_observations = replan_observations - 1

    all_observations = []
    labels = []
    all_observations.extend(initial_observations)
    labels.extend(['Non-reactive']*len(initial_observations))
    all_observations.extend(replan_observations)
    labels.extend(['Reactive']*len(replan_observations))
    print(len(initial_observations))
    print(sum(initial_observations))
    print(len(replan_observations))
    print(sum(replan_observations))
    all_data = {'Planner': labels,'Number of re-observations': all_observations}
    all_df = pd.DataFrame(data=all_data)
    number_of_reobs = []
    initial_reobs = []
    replan_reobs = []
    for i in range(int(np.max([np.max(replan_observations),np.max(initial_observations)]))+1):
        number_of_reobs.append(i)
        init_obs_count = 0
        for init_obs in initial_observations:
            if init_obs == i:
                init_obs_count += 1
        initial_reobs.append(init_obs_count)
        replan_obs_count = 0
        for replan_obs in replan_observations:
            if replan_obs == i:
                replan_obs_count += 1
        replan_reobs.append(replan_obs_count)

    plt.plot(number_of_reobs,initial_reobs,color="r",label="Non-reactive")
    plt.plot(number_of_reobs,replan_reobs,color="b",label="Reactive")
    plt.xlabel("Number of re-observations")
    plt.ylabel("Quantity")
    plt.legend()

    #sns.kdeplot(all_df,x='Number of re-observations',hue='Planner',palette=['red','blue'],clip=[0,20],bw_adjust=2)

    # xint = []
    # locs, labels = plt.xticks()
    # for each in locs:
    #     xint.append(int(each))
    # plt.xticks(xint)

    #plt.gca().set_xlim(left=0)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(plot_dir+"/"+row_key+"_decdec_linear_082024.png",dpi=300, bbox_inches="tight")

    plt.close()