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


def nearest(items, pivot):
    return min([i for i in items if i <= pivot], key=lambda x: abs(x - pivot))

data_crs = ccrs.PlateCarree()

plt.figure(figsize=(12, 6))
ax = plt.axes(projection=data_crs)
ax.set_global()
#ax.set_extent([-150, -30, 20, 70], crs=ccrs.PlateCarree())
# ax.set_xlim([-150,-30])
# ax.set_ylim([20,70])
x0c, x1c, y0c, y1c = ax.properties()['extent']
ax.coastlines()

# settings
var = 1
num_points_per_cell = 10



event_lats = []
event_lons = []
center_lats = np.arange(-85,95,10)
print(center_lats)
center_lons = np.arange(-175,185,10)
for clat in center_lats:
    for clon in center_lons:
        mean = [clat, clon]
        cov = [[var, 0], [0, var]]
        x, y = np.random.multivariate_normal(mean, cov, num_points_per_cell).T
        event_lats.extend(x)
        event_lons.extend(y)

plt.scatter(event_lons,event_lats,2,marker='o',color='blue',transform=data_crs)

# legend stuff
plt.scatter([], [], c='blue',marker='o', label='Grid location')


# Put a legend to the right of the current axis
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', fontsize=5, bbox_to_anchor=(1, 0.5))
# #m.imshow(precip, origin='upper', cmap='RdYlGn_r', vmin=1, vmax=200, zorder=3)

plt.savefig('./events.png',dpi=200)
plt.close()