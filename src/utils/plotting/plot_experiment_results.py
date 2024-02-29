import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd

metric_ind = 37 # 37 for het
rows = []
metrics = []
plot_dir = "./experiment_plots_grid/"
if not os.path.exists(plot_dir):
    os.mkdir("./experiment_plots_grid/")
# with open("./fire_parameter_study.csv",newline='') as csv_file:
#     spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

#     i = 0
#     for row in spamreader:
#         if i < 1:
#             i=i+1
#             continue
#         # for 1
#         # fov 2
#         # num_planes 3
#         # num_sats_per_plane 4
#         # agility 5
#         # event duration 6
#         # num_events 7
#         # event clustering 8
#         # init_events_obs_count 22
#         # init_events_1 24
#         # replan_events_1 38
#         # oracle_events_1 52
#         plt.figure()
#         title = row[0]+", FOR: "+row[1]+", FOV: "+row[2]+", n_p: "+row[3]+", n_sp: "+row[4]+", agility: "+row[5]+",\n event duration: "+row[6]+", event count: "+row[7]+", event clustering: "+row[8]
#         N = 5
#         ind = np.arange(N)  
#         width = 0.25
#         xvals = [int(row[22]), int(row[24]), int(row[25]), int(row[26]), int(row[27])] 
#         bar1 = plt.bar(ind, xvals, width, color = 'r') 
        
#         yvals = [int(row[36]), int(row[38]), int(row[39]), int(row[40]), int(row[41])] 
#         bar2 = plt.bar(ind+width, yvals, width, color='g') 
        
#         zvals = [int(row[50]), int(row[52]), int(row[53]), int(row[54]), int(row[55])] 
#         bar3 = plt.bar(ind+width*2, zvals, width, color = 'b') 
        
#         plt.xlabel("Number of observations per event") 
#         plt.ylabel('Observation count') 
#         plt.title(title)

#         plt.xticks(ind+width,['Total', '1', '2', '3', '4+']) 
#         plt.legend( (bar1, bar2, bar3), ('Initial', 'Reactive', 'Oracle') ) 
#         plt.savefig(plot_dir+row[0]+".png",dpi=300)
#         plt.close()
    
with open("./updated_grid_search.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        # for 1
        # fov 2
        # num_planes 3
        # num_sats_per_plane 4
        # agility 5
        # event duration 6
        # num_events 7
        # event clustering 8
        # init_events_obs_count 22
        # init_events_1 24
        # replan_events_1 38
        # oracle_events_1 52
       # plt.figure()
        fig, ax1 = plt.subplots()
        plt.rcParams.update({'font.size': 12})
        
        title = row[0]+", FOR: "+row[1]+", FOV: "+row[2]+", n_p: "+row[3]+", n_sp: "+row[4]+", agility: "+row[5]+",\n event duration: "+row[6]+", event count: "+row[7]+", event clustering: "+row[8]
        N = 4
        ind = np.arange(N)+1
        width = 0.25
        distribution = [1,2,3,4]
        xvals = [int(row[24]), int(row[25]), int(row[26]), int(row[27])]
        initial_observations = []
        initial_observations.extend([1]*int(row[24]))
        initial_observations.extend([2]*int(row[25]))
        initial_observations.extend([3]*int(row[26]))
        initial_observations.extend([4]*int(row[27]))
        x_data = {'count': initial_observations}
        x_df = pd.DataFrame(data=x_data)
        #sns.set_palette('theme')
        #sns.histplot(x_df,kde=True,palette=['red'],label='Initial')
        #sns.kdeplot(x_df,palette=['red'],label='Initial',ax=ax1)
        

        yvals = [int(row[38]), int(row[39]), int(row[40]), int(row[41])] 
        reactive_observations = []
        reactive_observations.extend([1]*int(row[38]))
        reactive_observations.extend([2]*int(row[39]))
        reactive_observations.extend([3]*int(row[40]))
        reactive_observations.extend([4]*int(row[41]))
        y_data = {'count': reactive_observations}
        y_df = pd.DataFrame(data=y_data)
        #sns.histplot(y_df,kde=True,palette=['blue'],label='Reactive')
        #sns.histplot(y_df,kde=True,color='b',label='Reactive')
        #sns.kdeplot(y_df,palette=['green'],label='Reactive',ax=ax1)
        
        
        zvals = [int(row[52]), int(row[53]), int(row[54]), int(row[55])] 
        oracle_observations = []
        oracle_observations.extend([1]*int(row[52]))
        oracle_observations.extend([2]*int(row[53]))
        oracle_observations.extend([3]*int(row[54]))
        oracle_observations.extend([4]*int(row[55]))
        z_data = {'count': oracle_observations}
        z_df = pd.DataFrame(data=z_data)
        #sns.histplot(z_df,kde=True,palette=['green'],label='Oracle')
        #sns.histplot(z_df,  color='g',label='Oracle')
        #sns.kdeplot(z_df,palette=['blue'],label='Oracle',ax=ax1)

        all_observations = []
        labels = []
        all_observations.extend(initial_observations)
        labels.extend(['Initial']*len(initial_observations))
        all_observations.extend(reactive_observations)
        labels.extend(['Reactive']*len(reactive_observations))
        # all_observations.extend(oracle_observations)
        # labels.extend(['Oracle']*len(oracle_observations))
        all_data = {'Planner': labels,'Number of re-observations': all_observations}
        all_df = pd.DataFrame(data=all_data)
        sns.kdeplot(all_df,x='Number of re-observations',hue='Planner',palette=['red','blue'],ax=ax1,clip=[1,4],bw_adjust=2)
        ax2 = ax1.twinx()
        if float(row[6]) == 6*3600:
            if float(row[3])*float(row[4]) > 4:
                ax2.set_ylim([0,250])
        bar1 = ax2.bar(ind-width/2, xvals, width, color='r',alpha=0.5) 
        bar2 = ax2.bar(ind+width/2, yvals, width, color='b',alpha=0.5) 
        #bar3 = ax2.bar(ind+width, zvals, width, color = 'b',alpha=0.5) 

        xint = []
        locs, labels = plt.xticks()
        for each in locs:
            xint.append(int(each))
        plt.xticks(xint)
        plt.xlim([0.5,4.5])

        #plt.xlabel("Number of observations per event") 
        plt.ylabel('Re-observation count') 
        #plt.title(title)

        #plt.xticks(ind+width,['Total', '1', '2', '3', '4+']) 
        #plt.legend( (hist1, hist2, hist3), ('Initial', 'Reactive', 'Oracle') ) 
        #plt.legend()
        plt.savefig(plot_dir+row[0]+"_hist.png",dpi=300)
        plt.close()