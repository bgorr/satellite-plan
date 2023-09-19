import csv
import numpy as np

filename = './events/lakes/lake_event_points_reduced.csv'

lake_locations = []
with open(filename,newline='') as csv_file:
    csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if i < 1:
            i=i+1
            continue
        row = [float(i) for i in row]
        lake_locations.append(row)

elapsed_time = 0
simulation_duration = 86400
simulation_step_size = 10
steps = np.arange(0,simulation_duration,simulation_step_size)

initial_requests = []

for lake in lake_locations:
    vis_event = [lake[0],lake[1],0,simulation_duration,1,["visible"]]
    sar_event = [lake[0],lake[1],0,simulation_duration,1,["sar"]]
    therm_event = [lake[0],lake[1],0,simulation_duration,1,["thermal"]]
    initial_requests.append(vis_event)
    initial_requests.append(sar_event)
    initial_requests.append(therm_event)

with open('./events/lakes/initial_requests_reduced.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity',"measurements"])
    for event in initial_requests:
        csvwriter.writerow(event)
