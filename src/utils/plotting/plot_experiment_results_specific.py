import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd

metric_ind = 37 # 37 for het
rows = []
metrics = []
plot_dir = "./experiment_plots_specific/"
if not os.path.exists(plot_dir):
    os.mkdir("./experiment_plots_specific/")
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

rows = {}
with open("./specific_reobservation_study.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        rows[row[0]] = row
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

#title = row[0]+", FOR: "+row[1]+", FOV: "+row[2]+", n_p: "+row[3]+", n_sp: "+row[4]+", agility: "+row[5]+",\n event duration: "+row[6]+", event count: "+row[7]+", event clustering: "+row[8]
N = 4
ind = np.arange(N)+1
print(ind)
width = 0.125
distribution = [1,2,3,4]
row = rows["specific_reobservation_default"]
xvals = [int(row[38]), int(row[39]), int(row[40]), int(row[41])] 
lin_inc_observations = []
lin_inc_observations.extend([1]*int(row[38]))
lin_inc_observations.extend([2]*int(row[39]))
lin_inc_observations.extend([3]*int(row[40]))
lin_inc_observations.extend([4]*int(row[41]))

row = rows["specific_reobservation_0"]
yvals = [int(row[38]), int(row[39]), int(row[40]), int(row[41])] 
lin_dec_observations = []
lin_dec_observations.extend([1]*int(row[38]))
lin_dec_observations.extend([2]*int(row[39]))
lin_dec_observations.extend([3]*int(row[40]))
lin_dec_observations.extend([4]*int(row[41]))

row = rows["specific_reobservation_1"]
zvals = [int(row[38]), int(row[39]), int(row[40]), int(row[41])] 
dec_dec_observations = []
dec_dec_observations.extend([1]*int(row[38]))
dec_dec_observations.extend([2]*int(row[39]))
dec_dec_observations.extend([3]*int(row[40]))
dec_dec_observations.extend([4]*int(row[41]))

row = rows["specific_reobservation_2"]
avals = [int(row[38]), int(row[39]), int(row[40]), int(row[41])] 
dec_inc_observations = []
dec_inc_observations.extend([1]*int(row[38]))
dec_inc_observations.extend([2]*int(row[39]))
dec_inc_observations.extend([3]*int(row[40]))
dec_inc_observations.extend([4]*int(row[41]))

row = rows["specific_reobservation_3"]
bvals = [int(row[38]), int(row[39]), int(row[40]), int(row[41])] 
imm_dec_observations = []
imm_dec_observations.extend([1]*int(row[38]))
imm_dec_observations.extend([2]*int(row[39]))
imm_dec_observations.extend([3]*int(row[40]))
imm_dec_observations.extend([4]*int(row[41]))

row = rows["specific_reobservation_4"]
cvals = [int(row[38]), int(row[39]), int(row[40]), int(row[41])] 
no_change_observations = []
no_change_observations.extend([1]*int(row[38]))
no_change_observations.extend([2]*int(row[39]))
no_change_observations.extend([3]*int(row[40]))
no_change_observations.extend([4]*int(row[41]))

all_observations = []
labels = []
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

# if float(row[6]) == 6*3600:
#     if float(row[3])*float(row[4]) > 4:
#         ax2.set_ylim([0,250])
#     else:
#         ax2.set_ylim([0,200])
bar1 = ax1.bar(ind-5*width/2, xvals, width, color='r',alpha=0.5)
bar2 = ax1.bar(ind-3*width/2, yvals, width, color='g',alpha=0.5)
bar3 = ax1.bar(ind-width/2, zvals, width, color = 'b',alpha=0.5)
bar4 = ax1.bar(ind+width/2, avals, width, color = 'yellow',alpha=0.5)
bar5 = ax1.bar(ind+3*width/2, bvals, width, color = 'orange',alpha=0.5)
bar6 = ax1.bar(ind+5*width/2, cvals, width, color = 'purple',alpha=0.5)
plt.ylabel('Re-observation count') 
ax2 = ax1.twinx()
sns.kdeplot(all_df,x='Number of re-observations',hue='Reobserving Strategy',palette=['red','green','blue','yellow','orange','purple'],ax=ax2,clip=[1,4],bw_adjust=2)

xint = []
locs, labels = plt.xticks()
for each in locs:
    xint.append(int(each))
plt.xticks(xint)
#plt.xlim([0.5,4.5])

#plt.xlabel("Number of observations per event") 

#plt.title(title)

#plt.xticks(ind+width,['Total', '1', '2', '3', '4+']) 
#plt.legend( (hist1, hist2, hist3), ('Initial', 'Reactive', 'Oracle') ) 
#plt.legend()
plt.savefig(plot_dir+"/reobserve_strategy_hist.png",dpi=300, bbox_inches="tight")

plt.close()