from SALib.sample import saltelli
from SALib.analyze import sobol, delta, pawn, ff, rbd_fast, hdmr
from SALib.test_functions import Ishigami
from SALib.plotting.bar import plot as barplot
from scipy.stats import norm, gaussian_kde, rankdata
from scipy.interpolate import UnivariateSpline
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import os

def compute_main_effects(feature_levels, feature_index, feature_array, metrics):
    levels_sum = 0
    for level in feature_levels:
        feature_present_sum = 0
        feature_not_present_sum = 0
        feature_present_count = 0
        feature_not_present_count = 0
        for i in range(len(feature_array[:,0])):
            if math.isclose(feature_array[i,feature_index],level):
                feature_present_sum += metrics[i]
                feature_present_count += 1
            else:
                feature_not_present_sum += metrics[i]
                feature_not_present_count += 1
        if feature_present_count == 0:
            print(feature_levels)
        if feature_not_present_count == 0:
            print(feature_levels)
        level_result = np.abs((1/feature_present_count)*feature_present_sum - (1/feature_not_present_count)*feature_not_present_sum)
        levels_sum += level_result
    return levels_sum/len(feature_levels)

def sobol_first(Y, X, m):
    # pre-process to catch constant array
    # see: https://github.com/numpy/numpy/issues/9631
    if Y.ptp() == 0.0:
        # Catch constant results
        # If Y does not change then it is not sensitive to anything...
        return 0.0
    xr = rankdata(X, method="ordinal")
    Vi = 0
    N = len(Y)
    Y_mean = Y.mean()
    for j in range(len(m) - 1):
        ix = np.where((xr > m[j]) & (xr <= m[j + 1]))[0]
        nm = len(ix)
        Vi += (nm / N) * ((Y[ix].mean() - Y_mean) ** 2)

    return Vi / np.var(Y)

def sobol_first_levels(Y, X, levels):
    # pre-process to catch constant array
    # see: https://github.com/numpy/numpy/issues/9631
    if Y.ptp() == 0.0:
        # Catch constant results
        # If Y does not change then it is not sensitive to anything...
        return 0.0
    Vi = 0
    N = len(Y)
    Y_mean = Y.mean()
    for j in range(len(levels) - 1):
        ix = np.where((X > levels[j]) & (X <= levels[j + 1]))[0]
        nm = len(ix)
        Vi += (nm / N) * ((Y[ix].mean() - Y_mean) ** 2)
    return Vi / np.var(Y)

def sobol_second_levels(Y, Xi, Xj, levels_i, levels_j):
    # pre-process to catch constant array
    # see: https://github.com/numpy/numpy/issues/9631
    if Y.ptp() == 0.0:
        # Catch constant results
        # If Y does not change then it is not sensitive to anything...
        return 0.0
    Vi = 0
    Vj = 0
    N = len(Y)
    Y_mean = Y.mean()
    for j in range(len(levels_i) - 1):
        ix = np.where((Xi > levels_i[j]) & (Xi <= levels_i[j + 1]))[0]
        nm = len(ix)
        Vi += (nm / N) * ((Y[ix].mean() - Y_mean) ** 2)
    for j in range(len(levels_j) - 1):
        ix = np.where((Xj > levels_j[j]) & (Xj <= levels_j[j + 1]))[0]
        nm = len(ix)
        Vj += (nm / N) * ((Y[ix].mean() - Y_mean) ** 2)
    Vij = 0
    for i in range(len(levels_i) - 1):
        for j in range(len(levels_j) - 1):
            ijx = np.where((Xi > levels_i[i]) & (Xi <= levels_i[i + 1]) & (Xj > levels_j[j]) & (Xj <= levels_j[j + 1]))[0]
            nm = len(ijx)
            Vij += (nm / N) * ((Y[ijx].mean() - Y_mean) ** 2)
    Vij = Vij - Vi - Vj
    return Vij / np.var(Y)

def sobol_first_conf(Y, X, m, num_resamples, conf_level, y_resamples):
    s = np.zeros(num_resamples)

    N = len(Y)
    r = np.random.randint(N, size=(num_resamples, y_resamples))

    for i in range(num_resamples):
        r_i = r[i, :]
        s[i] = sobol_first(Y[r_i], X[r_i], m)
    return norm.ppf(0.5 + conf_level / 2) * s.std(ddof=1)

def sobol_first_std(Y, X, m, num_resamples, conf_level, y_resamples):
    s = np.zeros(num_resamples)

    N = len(Y)
    r = np.random.randint(N, size=(num_resamples, y_resamples))

    for i in range(num_resamples):
        r_i = r[i, :]
        s[i] = sobol_first(Y[r_i], X[r_i], m)
    return s.std()
directory = "./src/utils/photos/experiments_032824/"
boxplot_directory = directory+"boxplots/"
if not os.path.exists(directory):
    os.mkdir(directory)
if not os.path.exists(boxplot_directory):
    os.mkdir(boxplot_directory)

# old formulation
# names = ['FOR', 'FOV', 'agility', 'event_duration', 'event_frequency', 'event_density', 'event_clustering', 'num_event_types']
# detailed_names = ['FOR (deg)', 'FOV (deg)', 'agility (deg/s)', 'event_duration (hrs)', 'event_frequency (events/hr/location)', 'event_density (events per 10 deg lat/lon)', 'event_clustering', 'num_event_types']
# feature_levels = [[5,10,30,60],
#                   [1,5,10,20],
#                   [0.01,0.1,1,5],
#                   [3600,6*3600,12*3600,24*3600],
#                   [2.78e-8,2.78e-7,2.78e-6,2.78e-5],
#                   [1,2,5,10],
#                   [1,4,8,16],
#                   [2,2,3,4]] 

names = ['FOR', 'FOV', 'Constellation', 'Agility', 'Event\nduration', 'Num\nevents', 'Event\nclustering']
detailed_names = ['FOR (deg)', 'FOV (deg)', 'constellation ID', 'agility (deg/s)', 'event_duration (hrs)', 'Num\nevents', 'Event\nclustering']
feature_levels = [[30,60],
                  [1,5,10],
                  [0,1,2,3],
                  [0.1,1,10],
                  [900,3600,3*3600,6*3600],
                  [1000,10000],
                  [0,1]
                  ]

# names = ['event_frequency', 'event_density']
# detailed_names = ['event_frequency (events/hr/location)', 'event_density (events per 10 deg lat/lon)']
# feature_levels = [[2.78e-8,2.78e-7,2.78e-6,2.78e-5],
#                   [1,2,5,10]]

# metric_dict = {
#     "Difference in co-obs percent coverage (all events)": 48,
#     "Difference in co-obs percent coverage (possible events)": 55,
#     "Difference in co-obs count": 49,
#     "Difference in average revisit": 50,
#     "Init percent coverage (all events)": 22,
#     "Replan percent coverage (all events)": 32,
#     "Init percent coverage (possible events)": 51,
#     "Replan percent coverage (possible events)": 52,
#     "Init co-obs count": 18,
#     "Replan co-obs count": 28,
#     "Init event maximum revisit": 23,
#     "Replan event maximum revisit": 33,
#     "Init event average revisit": 24,
#     "Replan event average revisit": 34,
#     "Difference in hom-het planning, co-obs count": 57,
#     "Difference in hom-het planning, percent coverage (possible events)": 54,
# }

metric_dict = {
    "Difference in co-obs percent coverage (all events)": 48,
    "Difference in co-obs percent coverage (possible events)": 55,
    "Difference in co-obs count": 49,
    "Difference in average revisit": 50,
    "Init percent coverage (all events)": 22,
    "Replan percent coverage (all events)": 32,
    "Init percent coverage (possible events)": 51,
    "Replan percent coverage (possible events)": 52,
    "Init co-obs count": 18,
    "Replan co-obs count": 28,
    "Init event maximum revisit": 23,
    "Replan event maximum revisit": 33,
    "Init event average revisit": 24,
    "Replan event average revisit": 34,
    "Difference in hom-het planning, co-obs count": 57,
    "Difference in hom-het planning, percent coverage (possible events)": 54,
}

metric_dict = { # new update
    #"Difference in reobs reward, oracle-replan": 63,
    "Difference in reobs reward, replan-init": 67,
    #"Difference in event count, oracle-replan": 65,
    "Difference in event count, replan-init": 69,
    #"Difference in obs event count, oracle-replan": 67,
    "Difference in obs event count, replan-init": 71
}

# metric_dict = { # new update, normalized
#     "Difference in reobs reward, oracle-replan": 72,
#     "Difference in reobs reward, replan-init": 73,
#     "Difference in event count, oracle-replan": 74,
#     "Difference in event count, replan-init": 75,
#     "Difference in obs event count, oracle-replan": 76,
#     "Difference in obs event count, replan-init": 77
# }

# for metric in metric_dict.keys():
#     rows = []
#     ys = []
#     with open("./updated_experiment.csv",newline='') as csv_file:
#         spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
#         i = 0
#         for row in spamreader:
#             if i < 1:
#                 i=i+1
#                 continue
#             ys.append(float(row[metric_dict[metric]]))
#             row = [float(i) for i in row[1:8]]
#             rows.append(row)
#     X = np.asarray(rows)

#     # Run model (example)
#     Y = np.transpose(np.asarray(ys))

#     S = {
#         'names': ['FOR', 'FOV', 'Constellation', 'Agility', 'Event\nduration', 'Num\nevents', 'Event\nclustering'],
#         "ME": np.zeros(shape=(7,1)),
#         "S1": np.zeros(shape=(7,1)),
#         "S1_conf": np.zeros(shape=(7,1)),
#         "S1_std": np.zeros(shape=(7,1))
#     }
#     num_resamples = 100
#     y_resamples = Y.size
#     exp = 2.0 / (7.0 + np.tanh((1500.0 - y_resamples) / 500.0))
#     M = int(np.round(min(int(np.ceil(y_resamples**exp)), 48)))
#     m = np.linspace(0, y_resamples, M+1)
#     conf_level = 0.95
#     for i in range(len(feature_levels)):
#         X_i = X[:, i]

#         ind = np.random.randint(Y.size, size=y_resamples)
#         S["ME"][i] = compute_main_effects(feature_levels[i], i, X, Y)
#         S["S1"][i] = sobol_first(Y[ind], X_i[ind], m)
#         S["S1"][i] = sobol_first_levels(Y, X_i, feature_levels[i])
#         S["S1_conf"][i] = sobol_first_conf(
#             Y, X_i, m, num_resamples, conf_level, y_resamples
#         )
#         S["S1_std"][i] = sobol_first_std(
#             Y, X_i, m, num_resamples, conf_level, y_resamples
#         )
#     for i in range(len(feature_levels)):
#         X_i = X[:, i]
#         for j in range(len(feature_levels)):
#             X_j = X[:,j]
#             if i != j:
#                 if sobol_second_levels(Y, X_i, X_j, feature_levels[i], feature_levels[j]) > 0.1:
#                     print(metric)
#                     print(names[i]+", "+names[j]+": "+str(sobol_second_levels(Y, X_i, X_j, feature_levels[i], feature_levels[j])))

#     ind = np.arange(len(S["names"]))
#     width = 0.35
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     rects1 = ax.bar(ind, np.squeeze(S["ME"]), width, color='royalblue')

#     # add some
#     ax.set_ylabel('Main effect')
#     #ax.set_title('Main effects on '+metric)
#     ax.set_xticks(ind)
#     ax.set_xticklabels( S["names"] )

#     plt.savefig(directory+metric+"_me.png",dpi=300,bbox_inches="tight")
#     plt.close()
#     #plt.show()

#     ind = np.arange(len(S["names"]))
#     width = 0.35
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     rects2 = ax.bar(ind, np.squeeze(S["S1"]), width, color='seagreen', yerr=np.squeeze(S["S1_std"]))
#     ax.set_ylim(0,0.5)
#     # add some
#     ax.set_ylabel('First-order sensitivity')
#     ax.set_title('Sensitivities for '+metric)
#     ax.set_xticks(ind)
#     ax.set_xticklabels( S["names"] )

#     plt.savefig(directory+metric+"_sobol.png",dpi=300,bbox_inches="tight")
#     plt.close()
#     #plt.show()

def heatmaps(var1_ind,var2_ind,var1_name,var2_name,metric_ind,metric_name,xscale_type,yscale_type):

    rows = []
    metrics = []
    with open("./grid_search_112623.csv",newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

        i = 0
        for row in spamreader:
            if i < 1:
                i=i+1
                continue
            metrics.append(float(row[metric_ind]))
            row = [float(i) for i in row[6:8]]
            row[0] = np.round(row[0]*3600,4)
            rows.append(row)

    X = np.asarray(rows)
    var1s = X[:, var1_ind]
    var2s = X[:, var2_ind]
    metrics = np.array(metrics)


    results = np.zeros(shape=(len(np.unique(var1s))*len(np.unique(var2s)),3))

    result_grid = np.zeros(shape=(len(np.unique(var1s)),len(np.unique(var2s))))
    ind = 0
    r = 0

    x_grid = np.zeros(shape=(len(np.unique(var1s)),len(np.unique(var2s))))
    y_grid = np.zeros(shape=(len(np.unique(var1s)),len(np.unique(var2s))))
    for var1 in np.unique(var1s):
        c = 0
        for var2 in np.unique(var2s):
            sum = 0
            count = 0
            for i in range(len(X[:,0])):
                if X[i, var1_ind] == var1 and X[i, var2_ind] == var2:
                    sum += metrics[i]
                    count += 1
            
            results[ind,0] = var1
            results[ind,1] = var2
            if count != 0:
                results[ind,2] = sum/count
            else:
                results[ind,2] = 0
            x_grid[r,c] = var1
            y_grid[r,c] = var2
            if count != 0:
                result_grid[r,c] = sum/count
            else:
                result_grid[r,c] = 0
            ind += 1
            c += 1
        r += 1

    fig, ax = plt.subplots()
    z_min, z_max = metrics.min(), metrics.max()
    plt.xscale(xscale_type)
    plt.yscale(yscale_type)
    xd = ax.scatter(results[:,0], results[:,1], c=results[:,2], cmap='Greens', vmin=np.abs(results[:,2]).min(), vmax=np.abs(results[:,2]).max())
    ax.set_title(metric_name+' scatter')

    ax.set_xlabel(detailed_names[names.index(var1_name)])
    ax.set_ylabel(detailed_names[names.index(var2_name)])
    fig.colorbar(xd, ax=ax, label=metric_name)
    plt.savefig(directory+var1_name+"_"+var2_name+"_"+metric_name+"_scatter.png")
    #plt.show()
    plt.close()

    z_min, z_max = -np.abs(result_grid).max(), np.abs(result_grid).max()

    fig, ax = plt.subplots()
    plt.xscale(xscale_type)
    plt.yscale(yscale_type)
    c = ax.pcolormesh(x_grid, y_grid, result_grid, cmap='Greens', vmin=np.abs(result_grid).min(), vmax=np.abs(result_grid).max())
    ax.set_title(metric_name+' heatmap')
    ax.set_xlabel(detailed_names[names.index(var1_name)])
    ax.set_ylabel(detailed_names[names.index(var2_name)])
    ax.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])

    fig.colorbar(c, ax=ax, label=metric_name)
    plt.savefig(directory+var1_name+"_"+var2_name+"_"+metric_name+"_heatmap.png")
    plt.close()
    #plt.show()
# for metric in metric_dict.keys():
#     for i in range(len(feature_levels)):
#         for j in range(len(feature_levels)):
#             if i > j:
#                 if names[i] in ['agility', 'event_frequency', 'event_density']:
#                     xscale_type = "log"
#                 else:
#                     xscale_type = "linear"
#                 if names[j] in ['agility', 'event_frequency', 'event_density']:
#                     yscale_type = "log"
#                 else:
#                     yscale_type = "linear"
#                 heatmaps(i,j,names[i],names[j],metric_dict[metric],metric,xscale_type,yscale_type)

# init_metric_ind = metric_dict['Init co-obs count']
# replan_metric_ind = metric_dict['Replan co-obs count']
# rows = []
# init_metrics = []
# replan_metrics = []
# with open("./grid_search_112623.csv",newline='') as csv_file:
#     spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

#     i = 0
#     for row in spamreader:
#         if i < 1:
#             i=i+1
#             continue
#         init_metrics.append(float(row[init_metric_ind]))
#         replan_metrics.append(float(row[replan_metric_ind]))
#         row = [float(i) for i in row[6:8]]
#         row[0] = np.round(row[0]*3600,4)
#         rows.append(row)
# for j in range(len(names)):
#     X = np.asarray(rows)
#     vars = X[:, j]
#     results = {}
#     for var in np.unique(vars):
#         init_metric_list = []
#         replan_metric_list = []
#         for i in range(len(X[:,0])):
#             if X[i, j] == var:
#                 init_metric_list.append(init_metrics[i])
#                 replan_metric_list.append(replan_metrics[i])
#         results[var] = {"init_metric_list": init_metric_list, "replan_metric_list": replan_metric_list}

#     fig, ax = plt.subplots()
#     box_results = []
#     box_replan_results = []
#     box_labels = []
#     box_replan_labels = []
#     for result in results.keys():
#         box_results.append(results[result]["init_metric_list"])
#         box_labels.append(str(result)+"_init")
#         box_results.append(results[result]["replan_metric_list"])
#         box_labels.append(str(result)+"_replan")
#     plt.xlabel(detailed_names[j]+", init and replan")
#     plt.ylabel("Co-obs count")
#     plt.ylim([0,1000])
#     plt.boxplot(box_results,labels=box_labels)
#     plt.savefig(boxplot_directory+"coobs_count_limited_"+str(names[j])+".png")
#     plt.close()
#     #plt.show()

# total_results = []
# total_results.append(init_metrics)
# total_results.append(replan_metrics)
# total_labels = ['Init','Replan']
# plt.xlabel("Init and replan")
# plt.ylabel("Co-obs count")
# plt.ylim([0,1000])
# plt.boxplot(total_results,labels=total_labels)
# plt.savefig(boxplot_directory+"coobs_count_limited_all.png")
# plt.close()
# #plt.show()

for metric in metric_dict.keys():
    rows = []
    diff_metric = []
    with open("./results/updated_experiment.csv",newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

        i = 0
        for row in spamreader:
            if i < 1:
                i=i+1
                continue
            diff_metric.append(float(row[metric_dict[metric]]))
            row = [float(i) for i in row[6:8]]
            row[0] = np.round(row[0]*3600,4)
            rows.append(row)

    fig, ax = plt.subplots()
    num_bins = 20
    n, bins, patches = ax.hist(diff_metric, num_bins, density=True,label="Histogram")
    sigma = np.std(diff_metric)
    mu = np.average(diff_metric)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
        np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    #plt.plot(bins,y,label="Fitted Gaussian")
    print(metric)
    print("Mean: ",np.mean(diff_metric))
    print("Max: ",np.max(diff_metric))
    print("Min: ",np.min(diff_metric))
    print("25th perc: ",np.percentile(diff_metric,25))
    print("50th perc: ",np.percentile(diff_metric,50))
    print("75th perc: ",np.percentile(diff_metric,75))
    sns.kdeplot(data=diff_metric,label="Kernel Density Estimation")
    plt.ylabel("Density")
    plt.xlabel(metric)
    plt.legend()
    plt.savefig(directory+metric+"_histogram_updated.png")