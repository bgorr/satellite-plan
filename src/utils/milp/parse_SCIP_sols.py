import re
import csv

# functions --------

# use regex to get tps from an observation : \\#(.*?)\\#(.*?)\\#(.*?)\\s
def extract_tps(scip_line):
    tps_array = re.findall("\d+",re.sub("(\s.+\s)\(obj:.+\)","",scip_line))
    return [tps_array[0],tps_array[1],tps_array[2]]

def read_solution_file(sol_file_name):
    tps_sol = []
    with open(sol_file_name, "r") as f:
        lines = f.readlines()

    i=0
    for line in lines:
        if re.search("^o#",line) != None:
            tps = extract_tps(line.strip())
            tps_sol.append(tps)
            i+=3
    return tps_sol
    
def write_plan(sol_file_name,access_file_name,output_filename):
    tps_sol = read_solution_file(sol_file_name)

    # find matching tps in access csv and use corresponding values to create csv file (for gifs/animations)
    obs = []
    with open(access_file_name,'r') as file:
        csvFile = csv.reader(file)
        
        for lines in csvFile:
            for tps in tps_sol:
                if tps == [lines[10], lines[8], lines[7]]:
                    obs.append(lines) 

    with open(output_filename,'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)

        csvWriter.writerows(obs)
    return obs
