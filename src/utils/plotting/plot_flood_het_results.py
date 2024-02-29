import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd

plot_dir = "./plot_flood_het_results/"
if not os.path.exists(plot_dir):
    os.mkdir("./plot_flood_het_results/")

rows = {}
with open("./flood_grid_search.csv",newline='') as csv_file:
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
        xvals = [int(row[42]), int(row[43]), int(row[44]), int(row[45])] # loose coobs, init
        initial_observations = []
        initial_observations.extend([1]*int(row[42]))
        initial_observations.extend([2]*int(row[43]))
        initial_observations.extend([3]*int(row[44]))
        initial_observations.extend([4]*int(row[45]))

        yvals = [int(row[66]), int(row[67]), int(row[68]), int(row[69])] 
        reactive_observations = []
        reactive_observations.extend([1]*int(row[66]))
        reactive_observations.extend([2]*int(row[67]))
        reactive_observations.extend([3]*int(row[68]))
        reactive_observations.extend([4]*int(row[69]))

        all_observations = []
        labels = []
        all_observations.extend(initial_observations)
        labels.extend(['Initial']*len(initial_observations))
        all_observations.extend(reactive_observations)
        labels.extend(['Reactive']*len(reactive_observations))

        all_data = {'Planner': labels,'Number of co-observations per event': all_observations}
        all_df = pd.DataFrame(data=all_data)

        bar1 = ax1.bar(ind-width/2, xvals, width, color='r',alpha=0.5) 
        bar2 = ax1.bar(ind+width/2, yvals, width, color = 'b',alpha=0.5)
        plt.ylabel('Co-observation count') 
        ax2 = ax1.twinx()
        sns.kdeplot(all_df,x='Number of co-observations per event',hue='Planner',palette=['red','blue'],ax=ax2,clip=[1,4])

        xint = []
        locs, labels = plt.xticks()
        for each in locs:
            xint.append(int(each))
        plt.xticks(xint)
        plt.xlim([0.5,4.5])

        plt.savefig(plot_dir+row[0]+"_hist.png",dpi=300,bbox_inches="tight")

        plt.close()