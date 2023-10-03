import csv
import numpy as np

bloom_filename = './events/lakes/bloom_events_reduced.csv'

events = []
with open(bloom_filename,newline='') as csv_file:
    csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if i < 1:
            i=i+1
            continue
        row = [float(i) for i in row]
        row.pop()
        row.append(2)
        events.append(row)

temp_filename = './events/lakes/temperature_events_reduced.csv'

with open(temp_filename,newline='') as csv_file:
    csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if i < 1:
            i=i+1
            continue
        row = [float(i) for i in row]
        row.pop()
        row.append(1)
        events.append(row)

level_filename = './events/lakes/level_events_reduced.csv'

with open(level_filename,newline='') as csv_file:
    csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
    i = 0
    for row in csvreader:
        if i < 1:
            i=i+1
            continue
        row = [float(i) for i in row]
        row.pop()
        row.append(0)
        events.append(row)

with open('./events/lakes/all_events_reduced.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['lat [deg]','lon [deg]','start time [s]','duration [s]', 'severity', 'measurements'])
    for event in events:
        csvwriter.writerow(event)
