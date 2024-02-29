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

plot_dir = "./plots/plot_flood_het_results/"
if not os.path.exists(plot_dir):
    os.mkdir("./plots/plot_flood_het_results/")
rows = {}
with open("./studies/flood_grid_search_het_twomeas.csv",newline='') as csv_file:
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
        width = 0.25
        distribution = [1,2,3,4]
        initial_observations = get_nonzero_observations(row[37])
        reactive_observations = get_nonzero_observations(row[53])
        reactive_het_observations = get_nonzero_observations(row[69])
        if len(initial_observations)==0 or len(reactive_observations)==0 or len(reactive_het_observations)==0:
            continue
        print(len(initial_observations))
        print(sum(initial_observations))
        print(len(reactive_observations))
        print(sum(reactive_observations))
        print(len(reactive_het_observations))
        print(sum(reactive_het_observations))
        all_observations = []
        labels = []
        all_observations.extend(initial_observations)
        labels.extend(['Initial']*len(initial_observations))
        all_observations.extend(reactive_observations)
        labels.extend(['Reactive']*len(reactive_observations))
        all_observations.extend(reactive_het_observations)
        labels.extend(['Reactive Het']*len(reactive_het_observations))
        all_data = {'Planner': labels,'Number of co-observations per event': all_observations}
        all_df = pd.DataFrame(data=all_data)

        sns.kdeplot(all_df,x='Number of co-observations per event',hue='Planner',palette=['red','blue','green'],clip=[1,10],bw_adjust=2,fill=True,alpha=.5, linewidth=0,)

        xint = []
        locs, labels = plt.xticks()
        for each in locs:
            xint.append(int(each))
        plt.xticks(xint)

        plt.savefig(plot_dir+row[0]+"_twomeas_hist.png",dpi=300,bbox_inches="tight")

        plt.close()