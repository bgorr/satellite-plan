import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd

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

with open("./studies/fire_constellation_study_lininc.csv",newline='') as csv_file:
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

    all_observations = []
    labels = []
    all_observations.extend(initial_observations)
    labels.extend(['Initial']*len(initial_observations))
    all_observations.extend(replan_observations)
    labels.extend(['Reactive']*len(replan_observations))
    print(len(initial_observations))
    print(sum(initial_observations))
    print(len(replan_observations))
    print(sum(replan_observations))
    all_data = {'Planner': labels,'Number of re-observations': all_observations}
    all_df = pd.DataFrame(data=all_data)


    sns.kdeplot(all_df,x='Number of re-observations',hue='Planner',palette=['red','blue'],clip=[1,20],bw_adjust=2)

    xint = []
    locs, labels = plt.xticks()
    for each in locs:
        xint.append(int(each))
    plt.xticks(xint)

    plt.savefig(plot_dir+"/"+row_key+"_lininc_hist.png",dpi=300, bbox_inches="tight")

    plt.close()