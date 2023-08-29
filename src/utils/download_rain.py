import subprocess
import datetime
import numpy as np

initial_time = datetime.datetime(2020,1,1,0,0,0)
steps = np.arange(0,864,1)
filenames = []
years = []
days = []
for step_num in steps:
    time = initial_time + datetime.timedelta(seconds=float(10*step_num))
    base_filename = "3B-HHR-L.MS.MRG.3IMERG.20200101-S000000-E002959.0000.V06B.HDF5"
    seconds_elapsed = 10*step_num
    half_hours_from_midnight = np.floor(seconds_elapsed / 1800)
    half_hours_in_minutes = int(30*half_hours_from_midnight)
    if (half_hours_from_midnight % 2) == 1:
        minutes_str = "30"
        end_minutes_str = "59"
    else:
        minutes_str = "00"
        end_minutes_str = "29"
    hours_str = str(int(np.floor(half_hours_from_midnight / 2)))
    filename = base_filename[0:23]+str(time.year)+str(time.month).zfill(2)+str(time.day).zfill(2)+"-S"+hours_str.zfill(2)+minutes_str+"00-E"+hours_str.zfill(2)+end_minutes_str+"59."+str(half_hours_in_minutes).zfill(4)+base_filename[52:]
    if filename not in filenames:
        filenames.append(filename)
        years.append(time.year)
        days.append((time-initial_time).days)

print(base_filename)
print(filenames)

for i in range(len(filenames)):    
    subprocess.run( ["wget","-P","/home/ben/repos/satplan/rain_data/", "--load-cookies", "/home/ben/.urs_cookies", "--save-cookies", "/home/ben/.urs_cookies", "--keep-session-cookies", "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHL.06/"+str(years[i])+"/"+str(days[i]+1).zfill(3)+"/"+filenames[i]])

# full base filepath: https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHL.06/2020/001/3B-HHR-L.MS.MRG.3IMERG.20200101-S000000-E002959.0000.V06B.HDF5