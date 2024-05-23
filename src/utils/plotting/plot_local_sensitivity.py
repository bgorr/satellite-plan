import csv
import matplotlib.pyplot as plt
import numpy as np

rows = []
constellations = []
duration_actuals = []
fors = []
fovs = []
agilities = []
num_events = []
clusterings = []
durations = []

diffs = []
percent_diffs = []
with open("./studies/local_sensitivity.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        row = [i for i in row]
        fors.append(int(row[1]))
        fovs.append(int(row[2]))
        agilities.append(float(row[5]))
        durations.append(int(row[6]))
        num_events.append(int(row[7]))
        clusterings.append(row[8])
        constellations.append((int(row[3]),int(row[4])))
        diffs.append(float(row[37])-float(row[22]))
        percent_diffs.append((float(row[37])-float(row[22]))/float(row[22]))
        rows.append(row)

print(fors)
print(diffs)
print(percent_diffs)
event_duration_options = [900/3600,3600/3600,3*3600/3600,6*3600/3600]
event_clustering_options = ["uniform","clustered"]
event_count_options = [1000,10000]
constellation_options = [(1,4),(2,2),(3,8),(8,3)]
agility_options = [0.1,1,10]
fov_options = [1,5,10]
for_options = [30,60]
parameters = {
    "num_events": event_count_options,
    "event_clustering": event_clustering_options,
    "event_duration": event_duration_options,
    "constellation_options": constellation_options,
    "max_slew_rate": agility_options,
    "ffor": for_options,
    "ffov": fov_options
}

i = 0
settings_list = []
#settings_list.append(default_settings)
for_default = 60
fov_default = 5
agility_default = 1.0
duration_default = 3*3600
num_event_default = 1000
clustering_default = "clustered"
constellation_default = (3,8)
for parameter in parameters:
    param_levels = []
    param_values = []
    for i in range(len(fors)):
        if parameter == "ffor" and fors[i] != for_default:
            print(fors[i])
            print(for_default)
            param_levels.append(str(fors[i]))
            param_values.append(percent_diffs[i]*100)
        if parameter == "ffov" and fovs[i] != fov_default:
            print(fovs[i])
            print(fov_default)
            param_levels.append(str(fovs[i]))
            param_values.append(percent_diffs[i]*100)
        if parameter == "max_slew_rate" and agilities[i] != agility_default:
            param_levels.append(str(agilities[i]))
            param_values.append(percent_diffs[i]*100)
        if parameter == "constellation_options" and constellations[i] != constellation_default:
            param_levels.append(str(constellations[i]))
            param_values.append(percent_diffs[i]*100)
        if parameter == "event_duration" and durations[i] != duration_default:
            param_levels.append(str(durations[i]/3600))
            param_values.append(percent_diffs[i]*100)
        if parameter == "num_events" and num_events[i] != num_event_default:
            param_levels.append(str(num_events[i]))
            param_values.append(percent_diffs[i]*100)
        if parameter == "event_clustering" and clusterings[i] != clustering_default:
            param_levels.append(str(clusterings[i]))
            param_values.append(percent_diffs[i]*100)
    if parameter == "ffor":
        param_levels.append(str(fors[-1]))
        param_values.append(percent_diffs[-1]*100)
    if parameter == "ffov":
        param_levels.append(str(fovs[-1]))
        param_values.append(percent_diffs[-1]*100)
    if parameter == "max_slew_rate":
        param_levels.append(str(agilities[-1]))
        param_values.append(percent_diffs[-1]*100)
    if parameter == "constellation_options":
        param_levels.append(str(constellations[-1]))
        param_values.append(percent_diffs[-1]*100)
    if parameter == "event_duration":
        param_levels.append(str(durations[-1]/3600))
        param_values.append(percent_diffs[-1]*100)
    if parameter == "num_events":
        param_levels.append(str(num_events[-1]))
        param_values.append(percent_diffs[-1]*100)
    if parameter == "event_clustering":
        param_levels.append(str(clusterings[-1]))
        param_values.append(percent_diffs[-1]*100)
    
    sorted_param_levels = []
    sorted_param_values = []
    label = ""
    if parameter == "ffor":
        label = "FOR (deg)"
        for f_o_r in for_options:
            param_ind = param_levels.index(str(f_o_r))
            sorted_param_levels.append(param_levels[param_ind])
            sorted_param_values.append(param_values[param_ind])
    if parameter == "ffov":
        label = "FOV (deg)"
        for f_o_v in fov_options:
            param_ind = param_levels.index(str(f_o_v))
            sorted_param_levels.append(param_levels[param_ind])
            sorted_param_values.append(param_values[param_ind])
    if parameter == "max_slew_rate":
        label = "Max slew rate (deg/s)"
        for agility in agility_options:
            param_ind = param_levels.index(str(float(agility)))
            sorted_param_levels.append(param_levels[param_ind])
            sorted_param_values.append(param_values[param_ind])
    if parameter == "event_duration":
        label = "Event duration (hrs)"
        for event_duration in event_duration_options:
            param_ind = param_levels.index(str(event_duration))
            sorted_param_levels.append(param_levels[param_ind])
            sorted_param_values.append(param_values[param_ind])
    if parameter == "event_clustering":
        label = "Event distribution"
        for event_clustering in event_clustering_options:
            param_ind = param_levels.index(str(event_clustering))
            sorted_param_levels.append(param_levels[param_ind])
            sorted_param_values.append(param_values[param_ind])
    if parameter == "num_events":
        label = "Number of events"
        for event_count in event_count_options:
            param_ind = param_levels.index(str(event_count))
            sorted_param_levels.append(param_levels[param_ind])
            sorted_param_values.append(param_values[param_ind])
    if parameter == "constellation_options":
        label = "Constellations ($n_p$,$n_s$)"
        for constellation in constellation_options:
            param_ind = param_levels.index(str(constellation))
            sorted_param_levels.append(param_levels[param_ind])
            sorted_param_values.append(param_values[param_ind])
    print(parameter)
    print(sorted_param_levels)
    print(sorted_param_values)
    plt.bar(sorted_param_levels,sorted_param_values)
    plt.xlabel(label,fontsize=12)
    plt.ylabel("Percent difference, reactive minus non-reactive",fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("./plots/"+parameter+"_sensitivity.png",dpi=300,bbox_inches="tight")
    plt.show()
        