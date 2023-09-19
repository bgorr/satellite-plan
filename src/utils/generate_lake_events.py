import csv
import numpy as np

filename = './events/lakes/lake_event_points_reduced.csv'

lake_locations = []
with open(filename,newline='') as csv_file:
    csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if i < 5:
            i=i+1
            continue
        row = [float(i) for i in row]
        lake_locations.append(row)

elapsed_time = 0
simulation_duration = 86400
event_types = ["bloom", "temperature", "level"]
event_durations = [6000, 3000, 1000]
event_frequencies = [0.0001,0.0002,0.0006]
simulation_step_size = 10
steps = np.arange(0,simulation_duration,simulation_step_size)

bloom_events = []
temperature_events = []
level_events = []


for step in steps:
    for lake in lake_locations:
        for i in range(len(event_types)):
            if np.random.random() < event_frequencies[i]:
                if event_types[i] == "bloom":
                    event = [lake[0],lake[1],step,event_durations[i],np.random.rand()*10,0]
                    bloom_events.append(event)
                if event_types[i] == "temperature":
                    event = [lake[0],lake[1],step,event_durations[i],np.random.rand()*10,1]
                    temperature_events.append(event)
                if event_types[i] == "level":
                    event = [lake[0],lake[1],step,event_durations[i],np.random.rand()*10,2]
                    level_events.append(event)

with open('./events/lakes/bloom_events_reduced.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity'])
    for event in bloom_events:
        csvwriter.writerow(event)

with open('./events/lakes/temperature_events_reduced.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity'])
    for event in temperature_events:
        csvwriter.writerow(event)

with open('./events/lakes/level_events_reduced.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]','severity'])
    for event in level_events:
        csvwriter.writerow(event)
