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
plot_dir = "./plots/constel_duration/"
if not os.path.exists(plot_dir):
    os.mkdir("./plots/constel_duration/")

with open("./studies/constel_duration_grid_search.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        rows[row[0]] = row

# fig, ax1 = plt.subplots()
# plt.rcParams.update({'font.size': 12})

# for row_key in rows.keys():
#     fig, ax1 = plt.subplots()
#     plt.rcParams.update({'font.size': 12})
#     row = rows[row_key]
#     print(row[23])
#     print(row[38])
#     initial_observations = get_nonzero_observations(row[23])
#     replan_observations = get_nonzero_observations(row[38])

#     initial_observations = initial_observations - 1
#     replan_observations = replan_observations - 1

#     all_observations = []
#     labels = []
#     all_observations.extend(initial_observations)
#     labels.extend(['Non-reactive']*len(initial_observations))
#     all_observations.extend(replan_observations)
#     labels.extend(['Reactive']*len(replan_observations))
#     print(len(initial_observations))
#     print(sum(initial_observations))
#     print(len(replan_observations))
#     print(sum(replan_observations))
#     all_data = {'Planner': labels,'Number of re-observations': all_observations}
#     all_df = pd.DataFrame(data=all_data)


#     sns.kdeplot(all_df,x='Number of re-observations',hue='Planner',palette=['red','blue'],clip=[0,20],bw_adjust=2)

#     xint = []
#     locs, labels = plt.xticks()
#     for each in locs:
#         xint.append(int(each))
#     plt.xticks(xint)

#     plt.gca().set_xlim(left=0)
#     plt.savefig(plot_dir+"/"+row_key+"_grid_search_hist_050124.png",dpi=300, bbox_inches="tight")

#     plt.close()

rows = []
planes = []
sats_per_plane = []
duration_actuals = []
diffs = []
percent_diffs = []
with open("./studies/constel_duration_grid_search.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        row = [i for i in row]
        planes.append(int(row[3]))
        sats_per_plane.append(int(row[4]))
        duration_actuals.append(int(row[6]))
        diffs.append(float(row[37])-float(row[22]))
        percent_diffs.append((float(row[37])-float(row[22]))/float(row[22]))
        #row[0] = np.round(row[0]*3600,4)
        rows.append(row)



# X = np.asarray(rows)
# planes = X[:, 3]
# sats_per_plane = X[:, 4]
# durations = X[:, 6]
# diffs = X[:,37] - X[:,22]
constellations = [(1,4),(2,2),(3,8),(8,3)]
constel_inds = [0,1,2,3]
duration_inds = [0,1,2,3,4,5]
durations = [900,3600,3*3600,6*3600,12*3600,24*3600]


results = np.zeros(shape=(len(constellations)*len(np.unique(durations)),3))

result_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
ind = 0
r = 0

x_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
y_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
for constel in constel_inds:
    c = 0
    for duration in duration_inds:
        sum = 0
        count = 0
        for i in range(len(planes)):
            if planes[i] == constellations[constel][0] and sats_per_plane[i] == constellations[constel][1] and duration_actuals[i] == durations[duration]:
                sum += diffs[i]
                count += 1
        
        results[ind,0] = constel
        results[ind,1] = duration
        if count != 0:
            results[ind,2] = sum/count
        else:
            results[ind,2] = 0
        x_grid[r,c] = constel
        y_grid[r,c] = duration
        if count != 0:
            result_grid[r,c] = sum/count
        else:
            result_grid[r,c] = 0
        ind += 1
        c += 1
    r += 1

z_min, z_max = -np.abs(result_grid).max(), np.abs(result_grid).max()

fig, ax = plt.subplots()
# plt.xscale(xscale_type)
# plt.yscale(yscale_type)
c = ax.pcolormesh(x_grid, y_grid, result_grid, cmap='Greens', vmin=np.abs(result_grid).min(), vmax=np.abs(result_grid).max())
a=ax.get_xticks().tolist()
a[0] = ''
a[1]='(1,4)'
a[2] = ''
a[3] = '(2,2)'
a[4] = ''
a[5] = '(3,8)'
a[6] = ''
a[7] = '(8,3)'
a[8] = ''
ax.set_xticklabels(a)

b=ax.get_yticks().tolist()

b[1]='15 min'
b[2] = '1 hr'
b[3] = '3 hrs'
b[4] = '6 hrs'
b[5] = '12 hrs'
b[6] = '24 hrs'
ax.set_yticklabels(b)
# ax.set_title(metric_name+' heatmap')
ax.set_xlabel('Constellation configuration ($n_p$,$n_s$)')
ax.set_ylabel('Event duration')
#ax.set_ylim(900,3600*6)
# ax.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])

fig.colorbar(c, ax=ax, label='Event observation count, reactive minus non-reactive')
plt.savefig("./plots/constel_duration_heatmap.png")
plt.close()

rows = {}
plot_dir = "./plots/constel_duration/"
if not os.path.exists(plot_dir):
    os.mkdir("./plots/constel_duration/")

constellations = [(1,4),(2,2),(3,8),(8,3)]
constel_inds = [0,1,2,3]
duration_inds = [0,1,2,3,4,5]
durations = [900,3600,3*3600,6*3600,12*3600,24*3600]


results = np.zeros(shape=(len(constellations)*len(np.unique(durations)),3))

result_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
ind = 0
r = 0

x_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
y_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
for constel in constel_inds:
    c = 0
    for duration in duration_inds:
        sum = 0
        count = 0
        for i in range(len(planes)):
            if planes[i] == constellations[constel][0] and sats_per_plane[i] == constellations[constel][1] and duration_actuals[i] == durations[duration]:
                sum += percent_diffs[i]*100
                count += 1
        
        results[ind,0] = constel
        results[ind,1] = duration
        if count != 0:
            results[ind,2] = sum/count
        else:
            results[ind,2] = 0
        x_grid[r,c] = constel
        y_grid[r,c] = duration
        if count != 0:
            result_grid[r,c] = sum/count
        else:
            result_grid[r,c] = 0
        ind += 1
        c += 1
    r += 1

z_min, z_max = -np.abs(result_grid).max(), np.abs(result_grid).max()

fig, ax = plt.subplots()
# plt.xscale(xscale_type)
# plt.yscale(yscale_type)
c = ax.pcolormesh(x_grid, y_grid, result_grid, cmap='Greens', vmin=np.abs(result_grid).min(), vmax=np.abs(result_grid).max())
a=ax.get_xticks().tolist()
a[0] = ''
a[1]='(1,4)'
a[2] = ''
a[3] = '(2,2)'
a[4] = ''
a[5] = '(3,8)'
a[6] = ''
a[7] = '(8,3)'
a[8] = ''
ax.set_xticklabels(a)

b=ax.get_yticks().tolist()
b[1]='15 min'
b[2] = '1 hr'
b[3] = '3 hrs'
b[4] = '6 hrs'
b[5] = '12 hrs'
b[6] = '24 hrs'
ax.set_yticklabels(b)
# ax.set_title(metric_name+' heatmap')
ax.set_xlabel('Constellation configuration ($n_p$,$n_s$)')
ax.set_ylabel('Event duration')
#ax.set_ylim(900,3600*6)
# ax.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])

fig.colorbar(c, ax=ax, label='Event observation count, percent improvement')
plt.savefig("./plots/constel_duration_heatmap_percent.png")
plt.close()

######################################################################################

rows = []
planes = []
sats_per_plane = []
duration_actuals = []
diffs = []
percent_diffs = []
with open("./results/updated_experiment.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        row = [i for i in row]
        planes.append(int(row[9]))
        sats_per_plane.append(int(row[10]))
        duration_actuals.append(int(row[5]))
        diffs.append(float(row[35])-float(row[20]))
        percent_diffs.append((float(row[35])-float(row[20]))/float(row[20]))
        rows.append(row)

constellations = [(1,4),(2,2),(3,8),(8,3)]
constel_inds = [0,1,2,3]
duration_inds = [0,1,2,3]
durations = [900,3600,3*3600,6*3600]


results = np.zeros(shape=(len(constellations)*len(np.unique(durations)),3))

result_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
ind = 0
r = 0

x_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
y_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
for constel in constel_inds:
    c = 0
    for duration in duration_inds:
        sum = 0
        count = 0
        for i in range(len(planes)):
            if planes[i] == constellations[constel][0] and sats_per_plane[i] == constellations[constel][1] and duration_actuals[i] == durations[duration]:
                sum += diffs[i]
                count += 1
        
        results[ind,0] = constel
        results[ind,1] = duration
        if count != 0:
            results[ind,2] = sum/count
        else:
            results[ind,2] = 0
        x_grid[r,c] = constel
        y_grid[r,c] = duration
        if count != 0:
            result_grid[r,c] = sum/count
        else:
            result_grid[r,c] = 0
        ind += 1
        c += 1
    r += 1

z_min, z_max = -np.abs(result_grid).max(), np.abs(result_grid).max()

fig, ax = plt.subplots()
# plt.xscale(xscale_type)
# plt.yscale(yscale_type)
c = ax.pcolormesh(x_grid, y_grid, result_grid, cmap='Greens', vmin=np.abs(result_grid).min(), vmax=np.abs(result_grid).max())
a=ax.get_xticks().tolist()
a[0] = ''
a[1]='(1,4)'
a[2] = ''
a[3] = '(2,2)'
a[4] = ''
a[5] = '(3,8)'
a[6] = ''
a[7] = '(8,3)'
a[8] = ''
ax.set_xticklabels(a)

b=ax.get_yticks().tolist()
b[0] = ''
b[1]='15 min'
b[2] = ''
b[3] = '1 hr'
b[4] = ''
b[5] = '3 hrs'
b[6] = ''
b[7] = '6 hrs'
b[8] = ''
ax.set_yticklabels(b)
# ax.set_title(metric_name+' heatmap')
ax.set_xlabel('Constellation configuration ($n_p$,$n_s$)')
ax.set_ylabel('Event duration')
#ax.set_ylim(900,3600*6)
# ax.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])

fig.colorbar(c, ax=ax, label='Event observation count, reactive minus non-reactive')
plt.savefig("./plots/constel_duration_heatmap_LHS.png")
plt.close()

rows = {}
plot_dir = "./plots/constel_duration/"
if not os.path.exists(plot_dir):
    os.mkdir("./plots/constel_duration/")

constellations = [(1,4),(2,2),(3,8),(8,3)]
constel_inds = [0,1,2,3]
duration_inds = [0,1,2,3]
durations = [900,3600,3*3600,6*3600]


results = np.zeros(shape=(len(constellations)*len(np.unique(durations)),3))

result_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
ind = 0
r = 0

x_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
y_grid = np.zeros(shape=(len(constellations),len(np.unique(durations))))
for constel in constel_inds:
    c = 0
    for duration in duration_inds:
        sum = 0
        count = 0
        for i in range(len(planes)):
            if planes[i] == constellations[constel][0] and sats_per_plane[i] == constellations[constel][1] and duration_actuals[i] == durations[duration]:
                sum += percent_diffs[i]
                count += 1
        
        results[ind,0] = constel
        results[ind,1] = duration
        if count != 0:
            results[ind,2] = sum/count
        else:
            results[ind,2] = 0
        x_grid[r,c] = constel
        y_grid[r,c] = duration
        if count != 0:
            result_grid[r,c] = sum/count
        else:
            result_grid[r,c] = 0
        ind += 1
        c += 1
    r += 1

z_min, z_max = -np.abs(result_grid).max(), np.abs(result_grid).max()

fig, ax = plt.subplots()
# plt.xscale(xscale_type)
# plt.yscale(yscale_type)
c = ax.pcolormesh(x_grid, y_grid, result_grid, cmap='Greens', vmin=np.abs(result_grid).min(), vmax=np.abs(result_grid).max())
a=ax.get_xticks().tolist()
a[0] = ''
a[1]='(1,4)'
a[2] = ''
a[3] = '(2,2)'
a[4] = ''
a[5] = '(3,8)'
a[6] = ''
a[7] = '(8,3)'
a[8] = ''
ax.set_xticklabels(a)

b=ax.get_yticks().tolist()
b[0] = ''
b[1]='15 min'
b[2] = ''
b[3] = '1 hr'
b[4] = ''
b[5] = '3 hrs'
b[6] = ''
b[7] = '6 hrs'
b[8] = ''
ax.set_yticklabels(b)
# ax.set_title(metric_name+' heatmap')
ax.set_xlabel('Constellation configuration ($n_p$,$n_s$)')
ax.set_ylabel('Event duration')
#ax.set_ylim(900,3600*6)
# ax.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])

fig.colorbar(c, ax=ax, label='Event observation count, percent improvement')
plt.savefig("./plots/constel_duration_heatmap_LHS_percent.png")
plt.close()