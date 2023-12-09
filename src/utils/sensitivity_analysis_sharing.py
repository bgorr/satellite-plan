import numpy as np
import matplotlib.pyplot as plt
import csv
import os

metric_ind = 37 # 37 for het
rows = []
metrics = []
with open("./grid_search_112823.csv",newline='') as csv_file:
    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')

    i = 0
    for row in spamreader:
        if i < 1:
            i=i+1
            continue
        metrics.append(float(row[metric_ind]))
        row = [float(i) for i in row[14:16]]
        rows.append(row)

X = np.asarray(rows)
names = ['sharing','planning']
var1_name = 'sharing'
var2_name = 'planning'
detailed_names = ['Sharing horizon (s)','Planning horizon (s)']
metric_name = 'Replan het co-obs count'
directory = './src/utils/photos/sharing/'
if not os.path.exists(directory):
    os.mkdir(directory)
var1_ind = 0
var2_ind = 1
var1s = X[:, 0]
var2s = X[:, 1]
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

z_min, z_max = -np.abs(result_grid).max(), np.abs(result_grid).max()

fig, ax = plt.subplots()
xscale_type = "log"
yscale_type = "log"
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