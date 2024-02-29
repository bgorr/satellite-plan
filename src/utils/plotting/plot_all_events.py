import time, calendar, datetime
from functools import partial
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import urllib, os
import csv
import numpy as np
import imageio
import sys
import multiprocessing
import h5py
import matplotlib.colors as colors
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
from multiprocessing import set_start_method
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader


def nearest(items, pivot):
    return min([i for i in items if i <= pivot], key=lambda x: abs(x - pivot))

data_crs = ccrs.PlateCarree()

plt.figure(figsize=(12, 6))
ax = plt.axes(projection=data_crs)
#ax.set_global()
#ax.set_xlim([-150,-30])
#ax.set_ylim([20,70])
x0c, x1c, y0c, y1c = ax.properties()['extent']
ax.set_extent([-125, -65, 25, 50], crs=ccrs.PlateCarree())
fname = './grwl_files/GRWL_summaryStats.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), edgecolor='lightskyblue', facecolor='None', linewidth=0.5)
ax.add_feature(shape_feature, edgecolor='lightskyblue', facecolor='None', linewidth=0.5)


ax.coastlines()
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray')

lightning_filepath = "./src/utils/lightning_events.csv"
fire_filepath = "./src/utils/fire_events.csv"
flood_filepath = "./src/utils/flow_events_75_updated.csv"

event_lats = []
event_lons = []
event_durations = []
with open(flood_filepath,'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if i < 1:
            i=i+1
            continue
        event_lats.append(float(row[0]))
        event_lons.append(float(row[1]))
        event_durations.append(float(row[3])/3600)
        # if float(row[3]) < 3600:
        #     event_durations.append(float(row[3])/60)

plt.scatter(event_lons,event_lats,0.5,marker='o',color='blue',transform=data_crs)

# legend stuff
plt.scatter([], [], c='red',marker='o', label='Lightning location')


# Put a legend to the right of the current axis
# #m.imshow(precip, origin='upper', cmap='RdYlGn_r', vmin=1, vmax=200, zorder=3)

plt.savefig('./flood_event_locations.png',dpi=300,bbox_inches="tight")
plt.close()

plt.hist(event_durations,color='blue')
plt.xlabel("Event duration (hours)")
plt.ylabel("Number of events")
#plt.xlim([0,3600])
plt.savefig('./flood_event_durations.png',dpi=300)
plt.close()