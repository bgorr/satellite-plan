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

plot_dir = "./plots/plot_flood_het_results/"
if not os.path.exists(plot_dir):
    os.mkdir("./plots/plot_flood_het_results/")
rows = {}
with open("./studies/flood_grid_search_het_threemeas.csv",newline='') as csv_file:
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
        initial_observations = get_nonzero_observations(row[38])
        reactive_observations = get_nonzero_observations(row[55])
        reactive_het_observations = get_nonzero_observations(row[72])
        if len(initial_observations)==0 or len(reactive_observations)==0 or len(reactive_het_observations)==0:
            continue
        initial_observations -= 1
        reactive_observations -= 1
        reactive_het_observations -= 1
        number_of_reobs = []
        initial_reobs = []
        reactive_reobs = []
        reactive_het_reobs = []
        for i in range(int(np.max([np.max(reactive_observations),np.max(initial_observations),np.max(reactive_het_observations)]))+1):
            number_of_reobs.append(i)
            init_obs_count = 0
            for init_obs in initial_observations:
                if init_obs == i:
                    init_obs_count += 1
            initial_reobs.append(init_obs_count)
            reactive_obs_count = 0
            for reactive_obs in reactive_observations:
                if reactive_obs == i:
                    reactive_obs_count += 1
            reactive_reobs.append(reactive_obs_count)
            reactive_het_obs_count = 0
            for reactive_het_obs in reactive_het_observations:
                if reactive_het_obs == i:
                    reactive_het_obs_count += 1
            reactive_het_reobs.append(reactive_het_obs_count)

        plt.plot(number_of_reobs,initial_reobs,color="red",linestyle=":",label="Non-reactive")
        plt.plot(number_of_reobs,reactive_reobs,color="blue",linestyle='--',label="Reactive")
        plt.plot(number_of_reobs,reactive_het_reobs,color="green",linestyle='-.',label="Reactive Het")
        plt.xlabel("Number of co-observations per event")
        plt.ylabel("Quantity")
        plt.legend()

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(plot_dir+row[0]+"_threemeas_linear_082024.png",dpi=300,bbox_inches="tight")

        plt.close()