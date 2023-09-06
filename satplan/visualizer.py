import os
import datetime
import csv
import math
import csv
import imageio
import multiprocessing

from mpl_toolkits.basemap import Basemap
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

class Visualizer:
    """
    # Visualizer Object

    Creates data 
    """
    def __init__(   self, 
                    data_dir : str, 
                    output_dir : str,
                    initial_datetime : datetime.datetime,
                    timestep : float,
                    duration : float,
                    coverage_grid_path : str = None
                ) -> None:
        """
        Creates an instance of a Visualizer object

        ### Arguments:
            - data_dir (`str`): directory pointing to the results from the simulation being visualized
            - output_dir (`str`): directory where the pre-computed data will be saved at
            - time (`datetime.datetime`): start date of the simulation
            - timestep (`float`): time-step used in the simulation in seconds [s]
            - duration (`float`): duarion of the simulation in seconds [s]
        """

        self.data_dir : str  = data_dir
        self.orbit_data_dir : str = data_dir + '/orbit_data/'
        self.coverage_grid_path : str = coverage_grid_path if coverage_grid_path else f'{data_dir}/coverage_grids/riverATLAS.csv'
        self.output_dir : str = output_dir
        self.initial_datetime : datetime.datetime = initial_datetime 
        self.timestep : float = timestep
        self.duration : float = duration

    def animate(self) -> None:
        print("Processing mission...")
        self.process_mission_data()
        print("Mission processed!")

        print("Generating animation...")
        self.plot_mission()
        print("Generating animation...")

    """
    --------------------------
        DATA PRE-PROCESSING
    --------------------------
    """
    def process_mission_data(self) -> None:
        """
        Pre-processes mission data for improved performance when animating.

        Saves processed data as a set of csv files saved to the desired output directory.
        """        
        # calculate julian date
        base_jd = self.__date_to_jd(self.initial_datetime.year,self.initial_datetime.month,self.initial_datetime.day) + (3600*self.initial_datetime.hour + 60*self.initial_datetime.minute + self.initial_datetime.second)/86400
        
        # Calculate contants
        steps = np.arange(0,self.duration*24*3600,self.timestep,dtype=int)
        conv = np.pi / (180.0*3600.0)
        xp   = -0.140682 * conv
        yp   =  0.333309 * conv
        lod  =  0.0015563
        ddpsi = -0.052195 * conv
        ddeps = -0.003875 * conv

        # Load satellite position, accesses, and measurements from simulation results
        satellites = self.__load_satellite_data()

        # Pre-process satellite position data
        self.__process_sat_position(steps, satellites, base_jd, lod, xp, yp, ddpsi, ddeps)       

        # Pre-process satellite visibility data
        self.__process_sat_visibility(steps, satellites, base_jd, lod, xp, yp, ddpsi, ddeps)

        # Pre-process satellite observations data
        self.__process_sat_obs(steps, satellites, base_jd, lod, xp, yp, ddpsi, ddeps)

        # Pre-process past observations 
        self.__process_prev_obs(steps, satellites, base_jd, lod, xp, yp, ddpsi, ddeps)

        # Pre-process ground swath data
        self.__process_ground_swaths(steps, satellites, base_jd, lod, xp, yp, ddpsi, ddeps)

        # Pre-process cross-link data
        self.__process_cross_links(steps, satellites, base_jd, lod, xp, yp, ddpsi, ddeps)


    def __load_satellite_data(self) -> list:
        satellites = []
        # for subdir in os.listdir(self.data_dir):

        for subdir in tqdm(
                        os.listdir(self.orbit_data_dir), 
                        desc="Loading satellite data", 
                        unit="sat_data"
                        ):
            satellite = {}
            
            if "comm" in subdir:
                continue
            
            if ".json" in subdir:
                continue
            
            if ".csv" in subdir:
                continue

            x = 1 
            for f in tqdm(
                        os.listdir(self.orbit_data_dir+subdir), 
                        desc=f"Loading {subdir} data", 
                        unit="files",
                        leave=False
                        ):           

                if "state_cartesian" in f:
                    with open(self.orbit_data_dir+subdir+"/"+f,newline='') as csv_file:
                        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                        states = []
                        i = 0
                        for row in spamreader:
                            if i < 5:
                                i=i+1
                                continue
                            row = [float(i) for i in row]
                            states.append(row)
                    satellite["orbitpy_id"] = subdir
                    satellite["states"] = states
                    
                if "access" in f:
                    with open(self.orbit_data_dir+subdir+"/"+f,newline='') as csv_file:
                        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                        visibilities = []
                        i = 0
                        for row in spamreader:
                            if i < 5:
                                i=i+1
                                continue
                            row = [float(i) for i in row]
                            visibilities.append(row)
                    satellite["visibilities"] = visibilities

                if "plan" in f:
                    with open(self.orbit_data_dir+subdir+"/"+f,newline='') as csv_file:
                        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                        observations = []
                        i = 0
                        for row in spamreader:
                            if i < 5:
                                i=i+1
                                continue
                            row = [float(i) for i in row]
                            observations.append(row)
                    satellite["observations"] = observations
                

            if "orbitpy_id" in satellite:
                satellites.append(satellite)

        return satellites


    def __process_sat_position(self, 
                                        steps : list, 
                                        satellites : list, 
                                        base_jd : float,
                                        lod : float,
                                        xp : float,
                                        yp : float,
                                        ddpsi : float, 
                                        ddeps : float) -> None:
        reprocess_data = False
        if not os.path.exists(self.output_dir+'/sat_positions'):
            # Create directory in ouput directory if needed
            os.mkdir(self.output_dir+'/sat_positions')
            reprocess_data = True
        else:
            # Check if pre-processed data is complete
            if len(os.listdir(self.output_dir+'/sat_positions')) != len(steps):
                # no match, clear directory
                for f in tqdm(
                            os.listdir(self.output_dir+'/sat_positions'), 
                            desc="Clearing processed satellite data", 
                            unit="files"
                            ):
                    if os.path.isdir(os.path.join(self.output_dir+'/sat_positions', f)):
                        for h in tqdm(
                            os.listdir(self.output_dir+'/sat_positions' + f), 
                            desc=f"Deleting files in `{self.output_dir+'/sat_positions'}/{f}`", 
                            unit="files"
                            ):
                            os.remove(os.path.join(self.output_dir+'/sat_positions', f, h))
                        os.rmdir(self.output_dir + f)

                    else:
                        os.remove(os.path.join(self.output_dir+'/sat_positions/', f)) 
                reprocess_data = True
                
        if reprocess_data:
            # pre-process data
            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing satellite position data", 
                            unit="files"
                        ):
                sat_positions = []

                for sat in satellites:
                    name = sat["orbitpy_id"]
                    states = sat["states"]
                    curr_state = states[i]
                    r_eci = [curr_state[1]*1e3,curr_state[2]*1e3,curr_state[3]*1e3]
                    v_eci = [curr_state[4]*1e3,curr_state[5]*1e3,curr_state[6]*1e3]
                    jd = base_jd + self.timestep*i/86400
                    centuries = (jd-2451545)/36525
                    r_ecef, _ = self.__eci2ecef(r_eci,v_eci,centuries,jd,lod,xp,yp,ddpsi,ddeps)
                    lat, lon, _ = self.__ecef2lla(r_ecef[0],r_ecef[1],r_ecef[2])
                    sat_position = [name,lat[0][0],lon[0][0]]
                    sat_positions.append(sat_position)


                with open(self.output_dir+'sat_positions/step'+str(i)+'.csv','w') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for pos in sat_positions:
                        csvwriter.writerow(pos)
        else:
            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing satellite position data", 
                            unit="files"
                        ):
                continue

    def __process_sat_visibility(self, 
                                steps : list, 
                                satellites : list, 
                                base_jd : float,
                                lod : float,
                                xp : float,
                                yp : float,
                                ddpsi : float, 
                                ddeps : float) -> None:
        reprocess_data = False
        if not os.path.exists(self.output_dir+'sat_visibilities'):
            os.mkdir(self.output_dir+'/sat_visibilities')
            reprocess_data = True
        else:
            # Check if pre-processed data is complete
            if len(os.listdir(self.output_dir+'/sat_visibilities')) != len(steps):
                # no match, clear directory
                for f in tqdm(
                            os.listdir(self.output_dir+'/sat_visibilities'), 
                            desc="Clearing processed satellite data", 
                            unit="files"
                            ):
                    if os.path.isdir(os.path.join(self.output_dir+'/sat_visibilities/', f)):
                        for h in tqdm(
                            os.listdir(self.output_dir+'/sat_visibilities' + f), 
                            desc=f"Deleting files in `{self.output_dir+'/sat_visibilities'}/{f}`", 
                            unit="files"
                            ):
                            os.remove(os.path.join(self.output_dir+'/sat_visibilities', f, h))
                        os.rmdir(self.output_dir + f)

                    else:
                        os.remove(os.path.join(self.output_dir+'/sat_visibilities/', f)) 
                reprocess_data = True
                
        if reprocess_data:
            # pre-process data
            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing satellite visibility data", 
                            unit="files"
                        ):
                sat_visibilities = []
                for sat in satellites:
                    name = sat["orbitpy_id"]
                    states = sat["states"]
                    curr_state = states[i]
                    r_eci = [curr_state[1]*1e3,curr_state[2]*1e3,curr_state[3]*1e3]
                    v_eci = [curr_state[4]*1e3,curr_state[5]*1e3,curr_state[6]*1e3]
                    jd = base_jd + self.timestep*i/86400
                    centuries = (jd-2451545)/36525
                    r_ecef, v_ecef = self.__eci2ecef(r_eci,v_eci,centuries,jd,lod,xp,yp,ddpsi,ddeps)
                    lat,lon,alt = self.__ecef2lla(r_ecef[0],r_ecef[1],r_ecef[2])
                    
                    for visibility in sat["visibilities"]:
                        if visibility[0] == i:
                            sat_pos_and_visibility = [name,lat[0][0],lon[0][0],visibility[2],visibility[3]]
                            sat_visibilities.append(sat_pos_and_visibility)

                with open(self.output_dir+'sat_visibilities/step'+str(i)+'.csv','w') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for vis in sat_visibilities:
                        csvwriter.writerow(vis)
        else:
            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing satellite visibility data", 
                            unit="files"
                        ):
                continue

    def __process_sat_obs(self, 
                                steps : list, 
                                satellites : list, 
                                base_jd : float,
                                lod : float,
                                xp : float,
                                yp : float,
                                ddpsi : float, 
                                ddeps : float) -> None:
        reprocess_data = False
        if not os.path.exists(self.output_dir+'sat_observations'):
            os.mkdir(self.output_dir+'sat_observations')
            reprocess_data = True
        else:
            # Check if pre-processed data is complete
            if len(os.listdir(self.output_dir+'/sat_observations')) != len(steps):
                # no match, clear directory
                for f in tqdm(
                            os.listdir(self.output_dir+'/sat_observations'), 
                            desc="Clearing processed satellite data", 
                            unit="files"
                            ):
                    if os.path.isdir(os.path.join(self.output_dir+'/sat_observations', f)):
                        for h in tqdm(
                            os.listdir(self.output_dir+'/sat_observations' + f), 
                            desc=f"Deleting files in `{self.output_dir+'/sat_observations'}/{f}`", 
                            unit="files"
                            ):
                            os.remove(os.path.join(self.output_dir+'/sat_observations', f, h))
                        os.rmdir(self.output_dir + '/sat_observations' + f)

                    else:
                        os.remove(os.path.join(self.output_dir+'/sat_observations', f)) 
                reprocess_data = True
                
        if reprocess_data:
            # pre-process data
            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing satellite observations data", 
                            unit="files"
                        ):
                sat_observations = []
                for sat in satellites:
                    name = sat["orbitpy_id"]
                    states = sat["states"]
                    curr_state = states[i]
                    r_eci = [curr_state[1]*1e3,curr_state[2]*1e3,curr_state[3]*1e3]
                    v_eci = [curr_state[4]*1e3,curr_state[5]*1e3,curr_state[6]*1e3]
                    jd = base_jd + self.timestep*i/86400
                    centuries = (jd-2451545)/36525
                    r_ecef, _ = self.__eci2ecef(r_eci,v_eci,centuries,jd,lod,xp,yp,ddpsi,ddeps)
                    lat, lon, _ = self.__ecef2lla(r_ecef[0],r_ecef[1],r_ecef[2])
                    
                    for observation in sat["observations"]:
                        if observation[0] <= i and i <= observation[1]:
                            sat_pos_and_observation = [name,lat[0][0],lon[0][0],observation[2],observation[3]]
                            sat_observations.append(sat_pos_and_observation)
                with open(self.output_dir+'sat_observations/step'+str(i)+'.csv','w') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for obs in sat_observations:
                        csvwriter.writerow(obs)
        else:
            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing satellite observations data", 
                            unit="files"
                        ):
                continue

    def __process_prev_obs(self, 
                                steps : list, 
                                satellites : list, 
                                base_jd : float,
                                lod : float,
                                xp : float,
                                yp : float,
                                ddpsi : float, 
                                ddeps : float) -> None:
        reprocess_data = False
        if not os.path.exists(self.output_dir+'/constellation_past_observations'):
            # Create directory in ouput directory if needed
            os.mkdir(self.output_dir+'constellation_past_observations')
            reprocess_data = True
        else:
            # Check if pre-processed data is complete
            if len(os.listdir(self.output_dir+'/constellation_past_observations')) != len(steps):
                # no match, clear directory
                for f in tqdm(
                            os.listdir(self.output_dir+'/constellation_past_observations'), 
                            desc="Clearing processed satellite data", 
                            unit="files"
                            ):
                    if os.path.isdir(os.path.join(self.output_dir+'/constellation_past_observations', f)):
                        for h in tqdm(
                            os.listdir(self.output_dir+'/constellation_past_observations' + f), 
                            desc=f"Deleting files in `{self.output_dir+'/constellation_past_observations'}/{f}`", 
                            unit="files"
                            ):
                            os.remove(os.path.join(self.output_dir+'/constellation_past_observations', f, h))
                        os.rmdir(self.output_dir + f)

                    else:
                        os.remove(os.path.join(self.output_dir+'/constellation_past_observations/', f)) 
                reprocess_data = True

        if reprocess_data:
            # pre-process data
            past_observations = []
            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing previous constellation observations data", 
                            unit="files"
                        ):
                for sat in satellites:
                    name = sat["orbitpy_id"]        
                    for observation in sat["observations"]:
                        if observation[0] == i:
                            prev_obs = None
                            for past_obs in past_observations:
                                if past_obs[1] == observation[2]:
                                    prev_obs = past_obs
                            if prev_obs is not None:
                                new_observation = [prev_obs[0]+1,observation[2],observation[3]]
                                past_observations.remove(prev_obs)
                            else:
                                new_observation = [1,observation[2],observation[3]]
                            past_observations.append(new_observation)
                with open(self.output_dir+'constellation_past_observations/step'+str(i)+'.csv','w') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for obs in past_observations:
                        csvwriter.writerow(obs)
        else:
            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing previous constellation observations data", 
                            unit="files"
                        ):
                continue

    def __process_ground_swaths(self, 
                                steps : list, 
                                satellites : list, 
                                base_jd : float,
                                lod : float,
                                xp : float,
                                yp : float,
                                ddpsi : float, 
                                ddeps : float) -> None:
        reprocess_data = False
        if not os.path.exists(self.output_dir+'/ground_swaths'):
            # Create directory in ouput directory if needed
            os.mkdir(self.output_dir+'ground_swaths')
            reprocess_data = True
        else:
            # Check if pre-processed data is complete
            if len(os.listdir(self.output_dir+'/ground_swaths')) != len(steps):
                # no match, clear directory
                for f in tqdm(
                            os.listdir(self.output_dir+'/ground_swaths'), 
                            desc="Clearing processed satellite data", 
                            unit="files"
                            ):
                    if os.path.isdir(os.path.join(self.output_dir+'/ground_swaths', f)):
                        for h in tqdm(
                            os.listdir(self.output_dir+'/ground_swaths' + f), 
                            desc=f"Deleting files in `{self.output_dir+'/ground_swaths'}/{f}`", 
                            unit="files"
                            ):
                            os.remove(os.path.join(self.output_dir+'/ground_swaths', f, h))
                        os.rmdir(self.output_dir + f)

                    else:
                        os.remove(os.path.join(self.output_dir+'/ground_swaths/', f)) 
                reprocess_data = True

        if reprocess_data:
            # pre-process data
            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing ground-swath data", 
                            unit="files"
                        ):
                ground_swath_points = []
                for sat in satellites:
                    name = sat["orbitpy_id"]
                    states = sat["states"]
                    curr_state = states[i]
                    r_eci = [curr_state[1]*1e3,curr_state[2]*1e3,curr_state[3]*1e3]
                    v_eci = [curr_state[4]*1e3,curr_state[5]*1e3,curr_state[6]*1e3]
                    jd = base_jd + self.timestep*i/86400
                    centuries = (jd-2451545)/36525
                    fov_points = [[-30,-30],[-30,30],[30,30],[30,-30]]
                    for fov_point in fov_points:
                        ground_point_eci = self.__pitchroll2ecisurface(r_eci,v_eci,fov_point[0],fov_point[1])
                        gp_ecef, _ = self.__eci2ecef(ground_point_eci,[0,0,0],centuries,jd,lod,xp,yp,ddpsi,ddeps)
                        gp_lat,gp_lon,_ = self.__ecef2lla(gp_ecef[0],gp_ecef[1],gp_ecef[2])
                        ground_swath_point = [name,gp_lat[0][0],gp_lon[0][0]]
                        ground_swath_points.append(ground_swath_point)
                with open(self.output_dir+'ground_swaths/step'+str(i)+'.csv','w') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for obs in ground_swath_points:
                        csvwriter.writerow(obs)
        else:
            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing ground-swath data", 
                            unit="files"
                        ):
                continue

    def __process_cross_links(self, 
                                steps : list, 
                                satellites : list, 
                                base_jd : float,
                                lod : float,
                                xp : float,
                                yp : float,
                                ddpsi : float, 
                                ddeps : float) -> None:

        reprocess_data = False
        if not os.path.exists(self.output_dir+'/crosslinks'):
            # Create directory in ouput directory if needed
            os.mkdir(self.output_dir+'crosslinks')
            reprocess_data = True
        else:
            # Check if pre-processed data is complete
            if len(os.listdir(self.output_dir+'/crosslinks')) != len(steps):
                # no match, clear directory
                for f in tqdm(
                            os.listdir(self.output_dir+'/crosslinks'), 
                            desc="Clearing processed satellite data", 
                            unit="files"
                            ):
                    if os.path.isdir(os.path.join(self.output_dir+'/crosslinks', f)):
                        for h in tqdm(
                            os.listdir(self.output_dir+'/crosslinks' + f), 
                            desc=f"Deleting files in `{self.output_dir+'/crosslinks'}/{f}`", 
                            unit="files"
                            ):
                            os.remove(os.path.join(self.output_dir+'/crosslinks', f, h))
                        os.rmdir(self.output_dir + f)

                    else:
                        os.remove(os.path.join(self.output_dir+'/crosslinks/', f)) 
                reprocess_data = True

        if reprocess_data:
            # pre-process data
            crosslinks = []
            for f in os.listdir(self.orbit_data_dir+"comm/"):
                csv_tokens = f.split("_")
                first_sat = csv_tokens[0]
                second_sat = csv_tokens[2][:-4]
                
                with open(self.orbit_data_dir+"comm/"+f,newline='') as csv_file:
                    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                    visibilities = []
                    i = 0
                    for row in spamreader:
                        if i < 4:
                            i=i+1
                            continue
                        row = [float(i) for i in row]
                        crosslink = [float(row[0]),float(row[1]),first_sat,second_sat]
                        crosslinks.append(crosslink)

            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing cross-links data", 
                            unit="files"
                        ):
                crosslink_locations = []     
                for crosslink in crosslinks:
                    lats = []
                    lons = []
                    if crosslink[0] <= i and i <= crosslink[1]:
                        for sat in satellites:
                            if sat["orbitpy_id"] == crosslink[2] or sat["orbitpy_id"] == crosslink[3]:
                                states = sat["states"]
                                curr_state = states[i]
                                r_eci = [curr_state[1]*1e3,curr_state[2]*1e3,curr_state[3]*1e3]
                                v_eci = [curr_state[4]*1e3,curr_state[5]*1e3,curr_state[6]*1e3]
                                jd = base_jd + self.timestep*i/86400
                                centuries = (jd-2451545)/36525
                                r_ecef, v_ecef = self.__eci2ecef(r_eci,v_eci,centuries,jd,lod,xp,yp,ddpsi,ddeps)
                                lat,lon,alt = self.__ecef2lla(r_ecef[0],r_ecef[1],r_ecef[2])
                                lats.append(lat)
                                lons.append(lon)
                        crosslink_location = [crosslink[2],crosslink[3],lats[0][0][0],lons[0][0][0],lats[1][0][0],lons[1][0][0]]
                        crosslink_locations.append(crosslink_location)
                with open(self.output_dir+'crosslinks/step'+str(i)+'.csv','w') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for obs in crosslink_locations:
                        csvwriter.writerow(obs)
        else:
            for i in tqdm(
                            range(len(steps)), 
                            desc="Processing cross-links data", 
                            unit="files"
                        ):
                continue


    def __precess(self, ttt):
        ttt2 = ttt*ttt
        ttt3 = ttt*ttt2
        # psia = np.deg2rad(5038.7784*ttt - 1.07259*ttt2 - 0.001147*ttt3)
        # wa   = np.deg2rad(84381.448                 + 0.05127*ttt2 - 0.007726*ttt3)
        # ea   = np.deg2rad(84381.448 -   46.8150*ttt - 0.00059*ttt2 + 0.001813*ttt3)
        # xa   = np.deg2rad(              10.5526*ttt - 2.38064*ttt2 - 0.001125*ttt3)
        zeta = np.deg2rad(            2306.2181*ttt + 0.30188*ttt2 + 0.017998*ttt3)/3600
        theta= np.deg2rad(            2004.3109*ttt - 0.42665*ttt2 - 0.041833*ttt3)/3600
        z    = np.deg2rad(            2306.2181*ttt + 1.09468*ttt2 + 0.018203*ttt3)/3600
        coszeta  = np.cos(zeta)
        sinzeta  = np.sin(zeta)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        cosz     = np.cos(z)
        sinz     = np.sin(z)
        prec = np.zeros(shape=(3,3))
        prec[0,0] =  coszeta * costheta * cosz - sinzeta * sinz
        prec[0,1] =  coszeta * costheta * sinz + sinzeta * cosz
        prec[0,2] =  coszeta * sintheta
        prec[1,0] = -sinzeta * costheta * cosz - coszeta * sinz
        prec[1,1] = -sinzeta * costheta * sinz + coszeta * cosz
        prec[1,2] = -sinzeta * sintheta
        prec[2,0] = -sintheta * cosz
        prec[2,1] = -sintheta * sinz
        prec[2,2] =  costheta
        return prec

    def __nutation(self, jd,ddpsi,ddeps):
        T = (jd-2451545)/36525
        l = self.__dms_to_dec(134,57,46.773) + (1325*180/np.pi+self.__dms_to_dec(198,52,2.633))*T + self.__dms_to_dec(0,0,31.310)*T**2 + self.__dms_to_dec(0,0,0.064)*T**3
        lp = self.__dms_to_dec(357,31,39.804) + (99*180/np.pi+self.__dms_to_dec(359,3,1.224))*T - self.__dms_to_dec(0,0,0.577)*T**2 - self.__dms_to_dec(0,0,0.012)*T**3
        F = self.__dms_to_dec(93,16,18.877) + (1342*180/np.pi+self.__dms_to_dec(82,1,3.137))*T - self.__dms_to_dec(0,0,13.257)*T**2 + self.__dms_to_dec(0,0,0.011)*T**3
        D = self.__dms_to_dec(297,51,1.307) + (1236*180/np.pi+self.__dms_to_dec(307,6,41.328))*T - self.__dms_to_dec(0,0,6.891)*T**2 + self.__dms_to_dec(0,0,0.019)*T**3
        Omega = self.__dms_to_dec(135,2,40.28) - (5*180/np.pi+self.__dms_to_dec(134,8,10.539))*T + self.__dms_to_dec(0,0,7.455)*T**2 + self.__dms_to_dec(0,0,0.008)*T**3

        with open(f'{self.orbit_data_dir}/iau80.csv',newline='') as csv_file:
            spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
            rows = []
            for row in spamreader:
                rows.append(row)
        del_psi = 0
        del_eps = 0
        for row in rows:
            A = float(row[0])*l+float(row[1])*lp+float(row[2])*F+float(row[3])*D+float(row[4])*Omega
            S = self.__dms_to_dec(0,0,1e-4*(float(row[5])+float(row[6])*T))
            C = self.__dms_to_dec(0,0,1e-4*(float(row[7])+float(row[8])*T))
            del_psi += np.deg2rad(S*np.sin(np.deg2rad(A)))
            del_eps += np.deg2rad(C*np.cos(np.deg2rad(A)))

        eps_0 = 84381.448 - self.__dms_to_dec(0,0,46.815)*T - self.__dms_to_dec(0,0,0.00059)*T**2 + self.__dms_to_dec(0,0,0.001813)*T**3
        eps_0 = np.deg2rad((eps_0/3600.0) % 360.0)
        del_psi =  (del_psi+ddpsi) % (2.0 * np.pi)
        del_eps =  (del_eps+ddeps) % (2.0 * np.pi)
        eps = eps_0 + del_eps
        nut = np.zeros(shape=(3,3))
        nut[0,0] = np.cos(del_psi)
        nut[0,1] = -np.sin(del_psi)*np.cos(eps_0)
        nut[0,2] = -np.sin(del_psi)*np.sin(eps_0)
        nut[1,0] = np.sin(del_psi)*np.cos(eps)
        nut[1,1] = np.cos(del_psi)*np.cos(eps)*np.cos(eps_0)+np.sin(eps)*np.sin(eps_0)
        nut[1,2] = np.cos(del_psi)*np.cos(eps)*np.sin(eps_0)-np.sin(eps)*np.cos(eps_0)
        nut[2,0] = np.sin(del_psi)*np.sin(eps)
        nut[2,1] = np.cos(del_psi)*np.sin(eps)*np.cos(eps_0)-np.cos(eps)*np.sin(eps_0)
        nut[2,2] = np.cos(del_psi)*np.sin(eps)*np.sin(eps_0)+np.cos(eps)*np.cos(eps_0)
        return del_psi, eps_0, np.deg2rad(Omega), nut

    def __gstime(self, jdut1):
        twopi      = 2.0*np.pi
        deg2rad    = np.pi/180.0

        tut1= ( jdut1 - 2451545.0 ) / 36525.0

        temp = - 6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1\
                + (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841
        temp = temp*deg2rad/240.0 % twopi

        if ( temp < 0.0 ):
            temp = temp + twopi
        gst = temp
        return gst


    def __sidereal(self, jdut1,deltapsi,meaneps,omega,lod):
        gmst= self.__gstime( jdut1 )
        if (jdut1 > 2450449.5 ):
            ast= gmst + deltapsi*np.cos(meaneps)\
                + 0.00264*np.pi /(3600*180)*np.sin(omega)\
                + 0.000063*np.pi /(3600*180)*np.sin(2.0 *omega)
        else:
            ast= gmst + deltapsi* np.cos(meaneps)

        ast = ast % (2.0*np.pi)
        thetasa    = 7.29211514670698e-05 * (1.0  - lod/86400.0 )
        omegaearth = thetasa
        st = np.zeros(shape=(3,3))
        st[0,0] = np.cos(ast)
        st[0,1] = -np.sin(ast)
        st[0,2] = 0.0
        st[1,0] = np.sin(ast)
        st[1,1] = np.cos(ast)
        st[1,2] = 0.0
        st[2,0] = 0.0
        st[2,1] = 0.0
        st[2,2] = 1.0

        stdot = np.zeros(shape=(3,3))
        stdot[0,0] = -omegaearth * np.sin(ast)
        stdot[0,1] = -omegaearth * np.cos(ast)
        stdot[0,2] = 0.0
        stdot[1,0] =  omegaearth * np.cos(ast)
        stdot[1,1] = -omegaearth * np.sin(ast)
        stdot[1,2] = 0.0
        stdot[2,0] = 0.0
        stdot[2,1] = 0.0
        stdot[2,2] = 0.0
        return st, stdot

    def __polarm(self, xp,yp):
        pm = np.zeros(shape=(3,3))
        pm[0,0] = np.cos(xp)
        pm[0,1] = 0
        pm[0,2] = -np.sin(xp)
        pm[1,0] = np.sin(xp)*np.sin(yp)
        pm[1,1] = np.cos(yp)
        pm[1,2] = np.cos(xp)*np.sin(yp)
        pm[2,0] = np.sin(xp)*np.cos(yp)
        pm[2,1] = -np.sin(yp)
        pm[2,2] = np.cos(xp)*np.cos(yp)
        return pm

    def __date_to_jd(self, year,month,day):
        """
        Convert a date to Julian Day.
        
        Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet', 
            4th ed., Duffet-Smith and Zwart, 2011.
        
        Parameters
        ----------
        year : int
            Year as integer. Years preceding 1 A.D. should be 0 or negative.
            The year before 1 A.D. is 0, 10 B.C. is year -9.
            
        month : int
            Month as integer, Jan = 1, Feb. = 2, etc.
        
        day : float
            Day, may contain fractional part.
        
        Returns
        -------
        jd : float
            Julian Day
            
        Examples
        --------
        Convert 6 a.m., February 17, 1985 to Julian Day
        
        >>> date_to_jd(1985,2,17.25)
        2446113.75
        
        """
        if month == 1 or month == 2:
            yearp = year - 1
            monthp = month + 12
        else:
            yearp = year
            monthp = month
        
        # this checks where we are in relation to October 15, 1582, the beginning
        # of the Gregorian calendar.
        if ((year < 1582) or
            (year == 1582 and month < 10) or
            (year == 1582 and month == 10 and day < 15)):
            # before start of Gregorian calendar
            B = 0
        else:
            # after start of Gregorian calendar
            A = math.trunc(yearp / 100.)
            B = 2 - A + math.trunc(A / 4.)
            
        if yearp < 0:
            C = math.trunc((365.25 * yearp) - 0.75)
        else:
            C = math.trunc(365.25 * yearp)
            
        D = math.trunc(30.6001 * (monthp + 1))
        
        jd = B + C + D + day + 1720994.5
        
        return jd

    def __dms_to_dec(self, d,m,s):
        return d + m/60 + s/3600

    def __eci2ecef(self, reci, veci, julian_centuries, jdut1, lod, xp, yp, ddpsi, ddeps):
        prec = self.__precess(julian_centuries)

        deltapsi,meaneps,omega,nut = self.__nutation(julian_centuries,ddpsi,ddeps)

        st,stdot = self.__sidereal(jdut1,deltapsi,meaneps,omega,lod)

        pm = self.__polarm(xp,yp)

        thetasa= 7.29211514670698e-05 * (1.0  - lod/86400.0)
        omegaearth = [0, 0, thetasa]

        r_pef = np.matmul(np.transpose(st),np.matmul(np.transpose(nut),np.matmul(np.transpose(prec),reci)))
        r_ecef = np.matmul(np.transpose(pm),r_pef)

        v_pef = np.matmul(np.transpose(st),np.matmul(np.transpose(nut),np.matmul(np.transpose(prec),veci))) - np.cross(omegaearth,r_pef)
        v_ecef = np.matmul(np.transpose(pm),v_pef)

        return r_ecef, v_ecef

    def __ecef2lla(self, x, y, z):
        # x, y and z are scalars or vectors in meters
        x = np.array([x]).reshape(np.array([x]).shape[-1], 1)
        y = np.array([y]).reshape(np.array([y]).shape[-1], 1)
        z = np.array([z]).reshape(np.array([z]).shape[-1], 1)

        a=6378137
        a_sq=a**2
        e = 8.181919084261345e-2
        e_sq = 6.69437999014e-3

        f = 1/298.257223563
        b = a*(1-f)

        # calculations:
        r = np.sqrt(x**2 + y**2)
        ep_sq  = (a**2-b**2)/b**2
        ee = (a**2-b**2)
        f = (54*b**2)*(z**2)
        g = r**2 + (1 - e_sq)*(z**2) - e_sq*ee*2
        c = (e_sq**2)*f*r**2/(g**3)
        s = (1 + c + np.sqrt(c**2 + 2*c))**(1/3.)
        p = f/(3.*(g**2)*(s + (1./s) + 1)**2)
        q = np.sqrt(1 + 2*p*e_sq**2)
        r_0 = -(p*e_sq*r)/(1+q) + np.sqrt(0.5*(a**2)*(1+(1./q)) - p*(z**2)*(1-e_sq)/(q*(1+q)) - 0.5*p*(r**2))
        u = np.sqrt((r - e_sq*r_0)**2 + z**2)
        v = np.sqrt((r - e_sq*r_0)**2 + (1 - e_sq)*z**2)
        z_0 = (b**2)*z/(a*v)
        h = u*(1 - b**2/(a*v))
        phi = np.arctan((z + ep_sq*z_0)/r)
        lambd = np.arctan2(y, x)


        return phi*180/np.pi, lambd*180/np.pi, h

    def __los_to_earth(self, position, pointing):
        """Find the intersection of a pointing vector with the Earth
        Finds the intersection of a pointing vector u and starting point s with the WGS-84 geoid
        Args:
            position (np.array): length 3 array defining the starting point location(s) in meters
            pointing (np.array): length 3 array defining the pointing vector(s) (must be a unit vector)
        Returns:
            np.array: length 3 defining the point(s) of intersection with the surface of the Earth in meters
        """

        a = 6378137.0
        b = 6378137.0
        c = 6356752.314245
        x = position[0]
        y = position[1]
        z = position[2]
        u = pointing[0]
        v = pointing[1]
        w = pointing[2]

        value = -a**2*b**2*w*z - a**2*c**2*v*y - b**2*c**2*u*x
        radical = a**2*b**2*w**2 + a**2*c**2*v**2 - a**2*v**2*z**2 + 2*a**2*v*w*y*z - a**2*w**2*y**2 + b**2*c**2*u**2 - b**2*u**2*z**2 + 2*b**2*u*w*x*z - b**2*w**2*x**2 - c**2*u**2*y**2 + 2*c**2*u*v*x*y - c**2*v**2*x**2
        magnitude = a**2*b**2*w**2 + a**2*c**2*v**2 + b**2*c**2*u**2

        if radical < 0:
            raise ValueError("The Line-of-Sight vector does not point toward the Earth")
        d = (value - a*b*c*np.sqrt(radical)) / magnitude

        if d < 0:
            raise ValueError("The Line-of-Sight vector does not point toward the Earth")

        return np.array([
            x + d * u,
            y + d * v,
            z + d * w,
        ])

    def __pitchroll2ecisurface(self, r_eci,v_eci, pitch, roll):
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
        yaw = np.deg2rad(0.0)
        z_hat_lvlh = r_eci/np.linalg.norm(r_eci) # from Earth center
        y_hat_lvlh = np.cross(r_eci,v_eci)/np.linalg.norm(np.cross(r_eci,v_eci)) # orbit normal
        x_hat_lvlh = np.cross(z_hat_lvlh,y_hat_lvlh) # complete the triad
        dcm_lvlh_to_eci = np.linalg.inv([[x_hat_lvlh[0],x_hat_lvlh[1],x_hat_lvlh[2]],
                        [y_hat_lvlh[0],y_hat_lvlh[1],y_hat_lvlh[2]],
                        [z_hat_lvlh[0],z_hat_lvlh[1],z_hat_lvlh[2]]])
        nadir_vec = [0,0,-1]
        roll_pitch_yaw = [[np.cos(yaw)*np.cos(pitch),np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll),np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                        [np.sin(yaw)*np.cos(pitch),np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll),np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                        [-np.sin(pitch),np.cos(pitch)*np.sin(roll),np.cos(pitch)*np.cos(roll)]]
        pointing_vec_lvlh = np.matmul(roll_pitch_yaw,nadir_vec)
        pointing_vec_eci = np.matmul(dcm_lvlh_to_eci,pointing_vec_lvlh)
        eci_intersection = self.__los_to_earth(r_eci,pointing_vec_eci)
        return eci_intersection

    """
    --------------------------
        ANIMATION
    --------------------------
    """
    def plot_step(self,step_num):
        filename = f'{self.output_dir}/plots/frame_{step_num}.png'

        m = Basemap(projection='merc',llcrnrlat=-75,urcrnrlat=75,\
                llcrnrlon=-180,urcrnrlon=180,resolution='c')
        pos_rows = []
        with open(self.output_dir+'/sat_positions/step'+str(step_num)+'.csv','r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                pos_rows.append(row)

        vis_rows = []
        with open(self.output_dir+'/sat_visibilities/step'+str(step_num)+'.csv','r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                vis_rows.append(row)

        obs_rows = []
        with open(self.output_dir+'/sat_observations/step'+str(step_num)+'.csv','r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                obs_rows.append(row)

        swath_rows = []
        with open(self.output_dir+'/ground_swaths/step'+str(step_num)+'.csv','r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                swath_rows.append(row)
        crosslinks = []
        with open(self.output_dir+'/crosslinks/step'+str(step_num)+'.csv','r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                crosslinks.append(row)

        past_lats = []
        past_lons = []
        past_rows = []
        with open(self.output_dir+'/constellation_past_observations/step'+str(step_num)+'.csv','r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                past_lats.append(float(row[1]))
                past_lons.append(float(row[2]))
                past_rows.append(row)

        # print grid
        grid_lats = []
        grid_lons = []
        with open(self.coverage_grid_path,'r') as csvfile:
            csvreader = csv.reader(csvfile,delimiter=',')
            next(csvfile)
            for row in csvreader:
                grid_lats.append(float(row[0]))
                grid_lons.append(float(row[1]))


        m.drawmapboundary(fill_color='#99ffff')
        m.fillcontinents(color='#cc9966',lake_color='#99ffff')
        time = self.initial_datetime
        time += datetime.timedelta(seconds=float(self.timestep*step_num))
        m.nightshade(time,alpha=0.1)
        ax = plt.gca()

        # fn = './plotting_data/3B-HHR-L.MS.MRG.3IMERG.20200101-S000000-E002959.0000.V06B.HDF5' #filename (the ".h5" file)
        # with h5py.File(fn) as f:      
        #     # retrieve image data:
        #     precip = f['/Grid/precipitationCal'][:]
        #     precip = np.flip( precip[0,:,:].transpose(), axis=0 )
        #     # get _FillValue for data masking
        #     #img_arr_fill = f[image].attrs['_FillValue'][0]   
        #     precip = np.ma.masked_less(precip, 1)
        #     precip = np.ma.masked_greater(precip, 200)
        
        x, y = m(grid_lons,grid_lats)
        m.scatter(x,y,2,marker='o',color='blue')

        x, y = m(past_lons,past_lats)
        m.scatter(x,y,3,marker='o',color='yellow')
        
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
        # m.imshow(precip, origin='upper', cmap='gray_r', vmin=0, vmax=10, zorder=3)
        plt.title('Simulation state at time t='+str(step_num)+' steps')
        plt.savefig(filename,dpi=300)
        plt.close()

    def plot_mission(self, plot_duration : float = 0.75):
        """ Creates an animated gif of the mission """

        # Check plot duration value
        if plot_duration < 0 or plot_duration > 1.0:
            raise ValueError('`plot_duration` can only be a value between [0,1]')

        # Set animation time
        steps = np.arange(int(np.floor(self.duration*plot_duration*86400/self.timestep)),int(np.floor(self.duration*86400/self.timestep)),1)
        filenames = [f'{self.output_dir}/plots/frame_{step}.png' for step in steps]

        # Generate Frames
        gen_frames = False
        if not os.path.exists(self.output_dir+'/plots/'):
            os.mkdir(self.output_dir+'/plots/')
            gen_frames = True
        elif len(os.listdir(self.output_dir + '/plots')) != len(steps):
            gen_frames = True
        else:
            dir_names = os.listdir(self.output_dir+'/plots/')
            gen_frames = False
            for filename in filenames:

                file_found = False
                for dir_name in dir_names:
                    file_found = filename in dir_name
                    if file_found:
                        break
                
                if file_found:
                    dir_names.remove(filename) 
                else:
                    gen_frames = True
                    break

        if gen_frames:
            # generate frames OF THE LAST 1/4th OF THE SIMULATION
            # imageio gif creation kills itself if there are too many images, is there a fix or is it just a WSL issue?
            pool = multiprocessing.Pool()
            for _ in tqdm(  pool.imap_unordered(self.plot_step, steps), 
                            total=len(steps),
                            desc='Geenrating plot frames',
                            unit='frames'):
                pass

        # Build GIF
        gif_name = self.output_dir+'/animation'
        with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print(f'Gif saved at {gif_name}\n')
