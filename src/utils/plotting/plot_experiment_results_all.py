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
results = []
with open("./results/updated_experiment.csv",newline='') as csv_file:
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
        
        title = row[0]+", FOR: "+row[1]+", FOV: "+row[2]+", n_p: "+row[3]+", n_sp: "+row[4]+", agility: "+row[5]+",\n event duration: "+row[6]+", event count: "+row[7]+", event clustering: "+row[8]
        result = {
            "event_duration": float(row[5]),
            "for": float(row[1]),
            "fov": float(row[2]),
            "agility": float(row[4]),
            "event_count": float(row[6]),
            "event_clustering": int(row[7]),
            "constellation": int(row[3]),
            "init_unique_event_count": float(row[21]),
            "init_event_obs_count": float(row[20]),
            "replan_unique_event_count": float(row[36]),
            "replan_event_obs_count": float(row[35])
        }
        results.append(result)

event_durations = [900,3600,10800,21600]
averages = []
stds = []
for event_duration in event_durations:
    sum = 0
    count = 0
    values = []
    for result in results:
        if result["event_duration"] == event_duration:
            sum += (result["replan_event_obs_count"]-result["init_event_obs_count"])
            values.append(result["replan_event_obs_count"]-result["init_event_obs_count"])
            count += 1
    averages.append(sum/count)
    stds.append(np.std(values))

event_durations = ["15 min","1 hour","3 hours","6 hours"]
plt.title("Difference in event obs count,\n averaged over the 49 LHS cases")
plt.bar(event_durations,averages)
plt.errorbar(event_durations,averages,yerr=stds, fmt="o", color="black",capsize=2.0)
plt.xlabel("Event durations")
plt.ylabel("Reactive minus non-reactive event obs count")
plt.savefig("./plots/event_duration_averages.png",dpi=300)
plt.show()

constellations = [0,1,3,2]
averages = []
stds = []
for constellation in constellations:
    sum = 0
    count = 0
    values = []
    for result in results:
        if result["constellation"] == constellation:
            sum += (result["replan_event_obs_count"]-result["init_event_obs_count"])
            values.append(result["replan_event_obs_count"]-result["init_event_obs_count"])
            count += 1
    averages.append(sum/count)
    stds.append(np.std(values))

constellations = ["(2,2)","(1,4)","(8,3)","(3,8)"]
plt.title("Difference in event obs count,\n averaged over the 49 LHS cases")
plt.bar(constellations,averages)
plt.errorbar(constellations,averages,yerr=stds, fmt="o", color="black",capsize=2.0)
plt.xlabel("Constellations ($n_p$,$n_s$)")
plt.ylabel("Reactive minus non-reactive event obs count")
plt.savefig("./plots/constellation_averages.png",dpi=300)
plt.show()

fors = [30,60]
averages = []
stds = []
for field_of_regard in fors:
    sum = 0
    count = 0
    values = []
    for result in results:
        if result["for"] == field_of_regard:
            sum += (result["replan_event_obs_count"]-result["init_event_obs_count"])
            values.append(result["replan_event_obs_count"]-result["init_event_obs_count"])
            count += 1
    averages.append(sum/count)
    stds.append(np.std(values))

fors = ["30","60"]
plt.title("Difference in event obs count,\n averaged over the 49 LHS cases")
plt.bar(fors,averages)
plt.errorbar(fors,averages,yerr=stds, fmt="o", color="black",capsize=2.0)
plt.xlabel("Constellations ($n_p$,$n_s$)")
plt.ylabel("Reactive minus non-reactive event obs count")
plt.savefig("./plots/for_averages.png",dpi=300)
plt.show()

fovs = [1,5,10]
averages = []
stds = []
for fov in fovs:
    sum = 0
    count = 0
    values = []
    for result in results:
        if result["fov"] == fov:
            sum += (result["replan_event_obs_count"]-result["init_event_obs_count"])
            values.append(result["replan_event_obs_count"]-result["init_event_obs_count"])
            count += 1
    averages.append(sum/count)
    stds.append(np.std(values))

fovs = ["1","5","10"]
plt.title("Difference in event obs count,\n averaged over the 49 LHS cases")
plt.bar(fovs,averages)
plt.errorbar(fovs,averages,yerr=stds, fmt="o", color="black",capsize=2.0)
plt.xlabel("Constellations ($n_p$,$n_s$)")
plt.ylabel("Reactive minus non-reactive event obs count")
plt.savefig("./plots/fov_averages.png",dpi=300)
plt.show()

event_counts = [1000,10000]
averages = []
stds = []
for event_count in event_counts:
    sum = 0
    count = 0
    values = []
    for result in results:
        if result["event_count"] == event_count:
            sum += (result["replan_event_obs_count"]-result["init_event_obs_count"])
            values.append(result["replan_event_obs_count"]-result["init_event_obs_count"])
            count += 1
    averages.append(sum/count)
    stds.append(np.std(values))

event_counts = ["1000","10000"]
plt.title("Difference in event obs count,\n averaged over the 49 LHS cases")
plt.bar(event_counts,averages)
plt.errorbar(event_counts,averages,yerr=stds, fmt="o", color="black",capsize=2.0)
plt.xlabel("Constellations ($n_p$,$n_s$)")
plt.ylabel("Reactive minus non-reactive event obs count")
plt.savefig("./plots/event_count_averages.png",dpi=300)
plt.show()

event_clusterings = [0,1]
averages = []
stds = []
for event_clustering in event_clusterings:
    sum = 0
    count = 0
    values = []
    for result in results:
        if result["event_clustering"] == event_clustering:
            sum += (result["replan_event_obs_count"]-result["init_event_obs_count"])
            values.append(result["replan_event_obs_count"]-result["init_event_obs_count"])
            count += 1
    averages.append(sum/count)
    stds.append(np.std(values))

event_clusterings = ["uniform","clustered"]
plt.title("Difference in event obs count,\n averaged over the 49 LHS cases")
plt.bar(event_clusterings,averages)
plt.errorbar(event_clusterings,averages,yerr=stds, fmt="o", color="black",capsize=2.0)
plt.xlabel("Constellations ($n_p$,$n_s$)")
plt.ylabel("Reactive minus non-reactive event obs count")
plt.savefig("./plots/event_clustering_averages.png",dpi=300)
plt.show()

agilities = [0.1,1,10]
averages = []
stds = []
for agility in agilities:
    sum = 0
    count = 0
    values = []
    for result in results:
        if result["agility"] == agility:
            sum += (result["replan_event_obs_count"]-result["init_event_obs_count"])
            values.append(result["replan_event_obs_count"]-result["init_event_obs_count"])
            count += 1
    averages.append(sum/count)
    stds.append(np.std(values))

agilities = ["0.1","1","10"]
plt.title("Difference in event obs count,\n averaged over the 49 LHS cases")
plt.bar(agilities,averages)
plt.errorbar(agilities,averages,yerr=stds, fmt="o", color="black",capsize=2.0)
plt.xlabel("Constellations ($n_p$,$n_s$)")
plt.ylabel("Reactive minus non-reactive event obs count")
plt.savefig("./plots/agility_averages.png",dpi=300)
plt.show()

sum = 0
count = 0
values = []
for result in results:
    sum += (result["replan_event_obs_count"]-result["init_event_obs_count"])/result["init_event_obs_count"]
    values.append((result["replan_event_obs_count"]-result["init_event_obs_count"])/result["init_event_obs_count"])
    count += 1

average = sum/count
print(average)
print(np.average(values))
print(np.std(values))
plt.hist(values,bins=20)
plt.xlabel("Percent improvement in event obs count, reactive vs. non-reactive")
plt.ylabel("Count")
plt.savefig("./plots/overall_hist.png",dpi=300,bbox_inches="tight")
plt.show()
