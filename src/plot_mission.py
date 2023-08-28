import time, calendar, datetime
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import urllib, os
import csv
import numpy as np
import imageio
import sys
import multiprocessing

def plot_step(step_num):
    filename = f'./missions/test_mission/plots/frame_{step_num}.png'
    m = Basemap(projection='merc',llcrnrlat=-75,urcrnrlat=75,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
    pos_rows = []
    with open('./missions/test_mission/sat_positions/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            pos_rows.append(row)

    vis_rows = []
    with open('./missions/test_mission/sat_visibilities/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            vis_rows.append(row)

    obs_rows = []
    with open('./missions/test_mission/sat_observations/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            obs_rows.append(row)

    swath_rows = []
    with open('./missions/test_mission/ground_swaths/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            swath_rows.append(row)
    crosslinks = []
    with open('./missions/test_mission/crosslinks/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            crosslinks.append(row)

    past_lats = []
    past_lons = []
    past_rows = []
    with open('./missions/test_mission/constellation_past_observations/step'+str(step_num)+'.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            past_lats.append(float(row[1]))
            past_lons.append(float(row[2]))
            past_rows.append(row)

    grid_lats = []
    grid_lons = []
    with open('./coverage_grids/riverATLAS.csv','r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        next(csvfile)
        for row in csvreader:
            grid_lats.append(float(row[0]))
            grid_lons.append(float(row[1]))


    m.drawmapboundary(fill_color='#99ffff')
    m.fillcontinents(color='#cc9966',lake_color='#99ffff')
    time = datetime.datetime(2020,1,1,0,0,0)
    time += datetime.timedelta(seconds=float(10*step_num))
    m.nightshade(time,alpha=0.25)

    x, y = m(grid_lons,grid_lats)
    m.scatter(x,y,2,marker='o',color='blue')

    x, y = m(past_lons,past_lats)
    m.scatter(x,y,3,marker='o',color='yellow')

    ax = plt.gca()
    for row in pos_rows:
        x, y = m(float(row[2]),float(row[1]))
        m.scatter(x,y,4,marker='^',color='black')
        ax.annotate(row[0], (x, y))
    
    for row in vis_rows:
        x, y = m(float(row[4]),float(row[3]))
        m.scatter(x,y,4,marker='o',color='orange')

    for row in obs_rows:
        obs_x, obs_y = m(float(row[4]),float(row[3]))
        #m.scatter(obs_x,obs_y,5,marker='o',color='green')
        sat_x, sat_y = m(float(row[2]),float(row[1]))
        if(np.sign(float(row[4])) != np.sign(float(row[2]))):
            continue
        xs = [obs_x,sat_x]
        ys = [obs_y,sat_y]
        m.plot(xs,ys,linewidth=1,color='r')

    for row in past_rows:
        if int(row[0]) > 1:
            x, y = m(float(row[2]),float(row[1]))
            ax.annotate(row[0], (x, y),fontsize=5)

    for row in crosslinks:
        sat1_x, sat1_y = m(float(row[3]),float(row[2]))
        sat2_x, sat2_y = m(float(row[5]),float(row[4]))
        if(np.sign(float(row[5])) != np.sign(float(row[3]))):
            continue
        xs = [sat1_x,sat2_x]
        ys = [sat1_y,sat2_y]
        m.plot(xs,ys,linewidth=0.5,linestyle='dashed',color='black')
        
    satlist = []
    for i in range(len(swath_rows)):
        if swath_rows[i][0] not in satlist:
            satlist.append(swath_rows[i][0])

    for sat in satlist:
        specific_sat = [x for x in swath_rows if sat == x[0]]
        xs = []
        ys = []
        longitude_sign = np.sign(float(specific_sat[0][2]))
        for row in specific_sat:
            x, y = m(float(row[2]),float(row[1]))
            if (not np.sign(float(row[2])) == longitude_sign) and (np.abs(float(row[2])) > 90.0):
                xs = []
                ys = []
                break
            xs.append(x)
            ys.append(y)
        x, y = m(float(specific_sat[0][2]),float(specific_sat[0][1]))
        xs.append(x)
        ys.append(y)
        m.plot(xs,ys,linewidth=0.5,color='purple')

    

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', fontsize=5, bbox_to_anchor=(1, 0.5))
    # plt.legend(fontsize=5,loc='upper right')
    plt.title('3D-CHESS observations at time t='+str(step_num)+' steps')
    plt.savefig(filename,dpi=300)
    plt.close()

pool = multiprocessing.Pool()
steps = np.arange(432,864,1)
pool.map(plot_step, steps)
filenames = []
for step in steps:
    filenames.append(f'./missions/test_mission/plots/frame_{step}.png')
# print('Charts saved\n')
gif_name = './missions/test_mission/animation'
# Build GIF
print('Creating gif\n')
with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
print('Gif saved\n')
# print('Removing Images\n')
# # Remove files
# for filename in set(filenames):
#     os.remove(filename)
# print('DONE')