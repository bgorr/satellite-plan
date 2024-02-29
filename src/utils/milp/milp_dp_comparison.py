import matplotlib.pyplot as plt
import csv

metric_ind = 37 # 37 for het
planner_ind = 10
event_density_ind = 7
agility_ind = 4
event_count_ind = 16
point_count_ind = 48
init_obs_count = 17
init_unique_event = 18
replan_obs_count = 27
replan_unique_event = 28
runtime_ind = 47

milp_init_xs = []
milp_init_ys = []
milp_replan_xs = []
milp_replan_ys = []
dp_init_xs = []
dp_init_ys = []
dp_replan_xs = []
dp_replan_ys = []

with open("./grid_search_120923.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        if float(row[agility_ind]) != 0.003:
            continue
        if float(row[event_density_ind]) > 5:
            continue
        if row[planner_ind] == "dp":
            dp_init_xs.append(float(row[event_density_ind]))
            dp_init_ys.append(float(row[init_unique_event]))
            dp_replan_xs.append(float(row[event_density_ind]))
            dp_replan_ys.append(float(row[replan_unique_event]))
        else:
            milp_init_xs.append(float(row[event_density_ind]))
            milp_init_ys.append(float(row[init_unique_event]))
            milp_replan_xs.append(float(row[event_density_ind]))
            milp_replan_ys.append(float(row[replan_unique_event]))

plt.plot(milp_init_xs,milp_init_ys,label="MILP static")
plt.plot(milp_replan_xs,milp_replan_ys,label="MILP reactive")
plt.plot(dp_init_xs,dp_init_ys,label="DP static")
plt.plot(dp_replan_xs,dp_replan_ys,label="DP reactive")

plt.xlabel("Event density", fontsize=12)
plt.ylabel("Number of unique events observed", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.savefig("milp_dp_comparison_unique_density.png",dpi=300)
plt.show()

milp_init_xs = []
milp_init_ys = []
milp_replan_xs = []
milp_replan_ys = []
dp_init_xs = []
dp_init_ys = []
dp_replan_xs = []
dp_replan_ys = []
with open("./grid_search_120923.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        if float(row[agility_ind]) != 0.003:
            continue
        if float(row[event_density_ind]) > 5:
            continue
        if row[planner_ind] == "dp":
            dp_init_xs.append(float(row[event_density_ind]))
            dp_init_ys.append(float(row[init_obs_count]))
            dp_replan_xs.append(float(row[event_density_ind]))
            dp_replan_ys.append(float(row[replan_obs_count]))
        else:
            milp_init_xs.append(float(row[event_density_ind]))
            milp_init_ys.append(float(row[init_obs_count]))
            milp_replan_xs.append(float(row[event_density_ind]))
            milp_replan_ys.append(float(row[replan_obs_count]))

plt.plot(milp_init_xs,milp_init_ys,label="MILP static")
plt.plot(milp_replan_xs,milp_replan_ys,label="MILP reactive")
plt.plot(dp_init_xs,dp_init_ys,label="DP static")
plt.plot(dp_replan_xs,dp_replan_ys,label="DP reactive")

plt.xlabel("Event density", fontsize=12)
plt.ylabel("Total number of observations of events", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.savefig("milp_dp_comparison_total_obs.png",dpi=300)
plt.show()

milp_init_xs = []
milp_init_ys = []
milp_replan_xs = []
milp_replan_ys = []
dp_init_xs = []
dp_init_ys = []
dp_replan_xs = []
dp_replan_ys = []
with open("./grid_search_120923.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        if float(row[agility_ind]) != 0.003:
            continue
        if row[planner_ind] == "dp":
            dp_init_xs.append(float(row[point_count_ind]))
            dp_init_ys.append(float(row[runtime_ind]))
        else:
            milp_init_xs.append(float(row[point_count_ind]))
            milp_init_ys.append(float(row[runtime_ind]))

plt.plot(milp_init_xs,milp_init_ys,label="MILP")
plt.plot(dp_init_xs,dp_init_ys,label="DP")

plt.yscale("log")
plt.xlabel("Number of grid points", fontsize=12)
plt.ylabel("Runtime (s)", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.savefig("milp_dp_comparison_runtime.png",dpi=300)
plt.show()

milp_init_xs = []
milp_init_ys = []
milp_replan_xs = []
milp_replan_ys = []
dp_init_xs = []
dp_init_ys = []
dp_replan_xs = []
dp_replan_ys = []
with open("./grid_search_120923.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        if float(row[event_density_ind]) != 5:
            continue
        if row[planner_ind] == "dp":
            dp_init_xs.append(float(row[agility_ind]))
            dp_init_ys.append(float(row[runtime_ind]))
        else:
            milp_init_xs.append(float(row[agility_ind]))
            milp_init_ys.append(float(row[runtime_ind]))

plt.plot(milp_init_xs,milp_init_ys,label="MILP")
plt.plot(dp_init_xs,dp_init_ys,label="DP")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Max torque (Nm)", fontsize=12)
plt.ylabel("Runtime (s)", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.savefig("milp_dp_comparison_runtime_agility.png",dpi=300)
plt.show()

bar_labels = ["MILP Static", "MILP Reactive", "DP Static", "DP Reactive"]
bar_counts = [247,243,243,242]
plt.bar(bar_labels,bar_counts)
plt.xlabel("Configuration", fontsize=12)
plt.ylabel("Unique events seen", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("milp_dp_comparison_old.png",dpi=300)
plt.show()