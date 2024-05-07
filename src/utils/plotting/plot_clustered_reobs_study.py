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

plot_dir = "./plots/plot_clustered_reobs/"
if not os.path.exists(plot_dir):
    os.mkdir("./plots/plot_clustered_reobs/")

rows = {}
with open("./studies/clustered_reobs_study.csv",newline='') as csv_file:
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
width = 0.125
distribution = [1,2,3,4]
row = rows["clustered_reobs_study"]
initial_observations = get_nonzero_observations(row[23])
lin_inc_observations = get_nonzero_observations(row[38])

row = rows["clustered_reobs_study_0"]
lin_dec_observations = get_nonzero_observations(row[38])

row = rows["clustered_reobs_study_1"]
dec_dec_observations = get_nonzero_observations(row[38])

row = rows["clustered_reobs_study_2"]
dec_inc_observations = get_nonzero_observations(row[38])

row = rows["clustered_reobs_study_3"]
imm_dec_observations = get_nonzero_observations(row[38])

row = rows["clustered_reobs_study_4"]
no_change_observations = get_nonzero_observations(row[38])

all_observations = []
labels = []
all_observations.extend(initial_observations)
labels.extend(['Non-reactive']*len(initial_observations))
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

p = sns.kdeplot(all_df,x='Number of re-observations',hue='Reobserving Strategy',palette=['black','red','green','blue','yellow','orange','purple'],clip=[1,10],bw_adjust=2, linestyle="--")

lss = [':', '--', '-.', '-', ':', '--', '-.']

handles = p.legend_.legendHandles[::-1]

for line, ls, handle in zip(p.lines, lss, handles):
    line.set_linestyle(ls)
    handle.set_ls(ls)

xint = []
locs, labels = plt.xticks()
for each in locs:
    xint.append(int(each))
plt.xticks(xint)

plt.savefig(plot_dir+"/clustered_reobs_hist.png",dpi=300, bbox_inches="tight")

plt.close()