import numpy as np
import datetime
import csv
import math
import os

def precess(ttt):
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

def nutation(jd,ddpsi,ddeps):
    T = (jd-2451545)/36525
    l = dms_to_dec(134,57,46.773) + (1325*180/np.pi+dms_to_dec(198,52,2.633))*T + dms_to_dec(0,0,31.310)*T**2 + dms_to_dec(0,0,0.064)*T**3
    lp = dms_to_dec(357,31,39.804) + (99*180/np.pi+dms_to_dec(359,3,1.224))*T - dms_to_dec(0,0,0.577)*T**2 - dms_to_dec(0,0,0.012)*T**3
    F = dms_to_dec(93,16,18.877) + (1342*180/np.pi+dms_to_dec(82,1,3.137))*T - dms_to_dec(0,0,13.257)*T**2 + dms_to_dec(0,0,0.011)*T**3
    D = dms_to_dec(297,51,1.307) + (1236*180/np.pi+dms_to_dec(307,6,41.328))*T - dms_to_dec(0,0,6.891)*T**2 + dms_to_dec(0,0,0.019)*T**3
    Omega = dms_to_dec(135,2,40.28) - (5*180/np.pi+dms_to_dec(134,8,10.539))*T + dms_to_dec(0,0,7.455)*T**2 + dms_to_dec(0,0,0.008)*T**3

    with open('./iau80.csv',newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
        rows = []
        for row in spamreader:
            rows.append(row)
    del_psi = 0
    del_eps = 0
    for row in rows:
        A = float(row[0])*l+float(row[1])*lp+float(row[2])*F+float(row[3])*D+float(row[4])*Omega
        S = dms_to_dec(0,0,1e-4*(float(row[5])+float(row[6])*T))
        C = dms_to_dec(0,0,1e-4*(float(row[7])+float(row[8])*T))
        del_psi += np.deg2rad(S*np.sin(np.deg2rad(A)))
        del_eps += np.deg2rad(C*np.cos(np.deg2rad(A)))

    eps_0 = 84381.448 - dms_to_dec(0,0,46.815)*T - dms_to_dec(0,0,0.00059)*T**2 + dms_to_dec(0,0,0.001813)*T**3
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

def gstime(jdut1):
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


def sidereal(jdut1,deltapsi,meaneps,omega,lod):
    gmst= gstime( jdut1 )
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

def polarm(xp,yp):
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

def date_to_jd(year,month,day):
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

def dms_to_dec(d,m,s):
    return d + m/60 + s/3600

def eci2ecef( reci, veci, julian_centuries, jdut1, lod, xp, yp, ddpsi, ddeps):
    prec = precess(julian_centuries)

    deltapsi,meaneps,omega,nut = nutation(julian_centuries,ddpsi,ddeps)

    st,stdot = sidereal(jdut1,deltapsi,meaneps,omega,lod)

    pm = polarm(xp,yp)

    thetasa= 7.29211514670698e-05 * (1.0  - lod/86400.0)
    omegaearth = [0, 0, thetasa]

    r_pef = np.matmul(np.transpose(st),np.matmul(np.transpose(nut),np.matmul(np.transpose(prec),reci)))
    r_ecef = np.matmul(np.transpose(pm),r_pef)

    v_pef = np.matmul(np.transpose(st),np.matmul(np.transpose(nut),np.matmul(np.transpose(prec),veci))) - np.cross(omegaearth,r_pef)
    v_ecef = np.matmul(np.transpose(pm),v_pef)

    return r_ecef, v_ecef

def ecef2lla(x, y, z):
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

def los_to_earth(position, pointing):
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

def pitchroll2ecisurface(r_eci,v_eci, pitch, roll):
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
    eci_intersection = los_to_earth(r_eci,pointing_vec_eci)
    return eci_intersection

    

def process_mission(settings):
    print("Processing mission")
    time = settings["initial_datetime"]
    base_directory = settings["directory"]
    #time = datetime.datetime(2020,1,1,0,0,0)
    base_jd = date_to_jd(time.year,time.month,time.day) + (3600*time.hour + 60*time.minute + time.second)/86400
    conv = np.pi / (180.0*3600.0)
    xp   = -0.140682 * conv
    yp   =  0.333309 * conv
    lod  =  0.0015563
    ddpsi = -0.052195 * conv
    ddeps = -0.003875 * conv

    directory = base_directory+"orbit_data/"

    satellites = []

    for subdir in os.listdir(directory):
        satellite = {}
        if "comm" in subdir:
            continue
        if ".json" in subdir:
            continue
        if ".csv" in subdir:
            continue
        for f in os.listdir(directory+subdir):
            
            if "state_cartesian" in f:
                with open(directory+subdir+"/"+f,newline='') as csv_file:
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
                with open(directory+subdir+"/"+f,newline='') as csv_file:
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
                with open(directory+subdir+"/"+f,newline='') as csv_file:
                    spamreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                    observations = []
                    i = 0
                    for row in spamreader:
                        if i < 1:
                            i=i+1
                            continue
                        row = [float(i) for i in row]
                        observations.append(row)
                satellite["observations"] = observations
            
        if settings["preplanned_observations"] is not None:
            with open(settings["preplanned_observations"],newline='') as csv_file:
                csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                observations = []
                i = 0
                for row in csvreader:
                    if i < 1:
                        i=i+1
                        continue
                    if int(row[0][8:]) == int(satellite["orbitpy_id"][3]):
                        obs = [int(float(row[3])),int(float(row[4])),float(row[1])*180/np.pi, float(row[2])*180/np.pi]
                        observations.append(obs)
            satellite["observations"] = observations
            

        if "orbitpy_id" in satellite:
            satellites.append(satellite)

    timestep = settings["step_size"]
    duration = settings["duration"]*86400
    steps = np.arange(0,duration,timestep,dtype=int)
    if not settings["process_obs_only"]:
        if not os.path.exists(base_directory+'sat_positions'):
            os.mkdir(base_directory+'sat_positions')
        for i in range(len(steps)):
            sat_positions = []
            for sat in satellites:
                name = sat["orbitpy_id"]
                states = sat["states"]
                curr_state = states[i]
                r_eci = [curr_state[1]*1e3,curr_state[2]*1e3,curr_state[3]*1e3]
                v_eci = [curr_state[4]*1e3,curr_state[5]*1e3,curr_state[6]*1e3]
                jd = base_jd + timestep*i/86400
                centuries = (jd-2451545)/36525
                r_ecef, v_ecef = eci2ecef(r_eci,v_eci,centuries,jd,lod,xp,yp,ddpsi,ddeps)
                lat,lon,alt = ecef2lla(r_ecef[0],r_ecef[1],r_ecef[2])
                sat_position = [name,lat[0][0],lon[0][0]]
                sat_positions.append(sat_position)
            with open(base_directory+'sat_positions/step'+str(i)+'.csv','w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for pos in sat_positions:
                    csvwriter.writerow(pos)

    if not settings["process_obs_only"]:
        if not os.path.exists(base_directory+'sat_visibilities'):
            os.mkdir(base_directory+'sat_visibilities')
        for i in range(len(steps)):
            sat_visibilities = []
            for sat in satellites:
                name = sat["orbitpy_id"]
                states = sat["states"]
                curr_state = states[i]
                r_eci = [curr_state[1]*1e3,curr_state[2]*1e3,curr_state[3]*1e3]
                v_eci = [curr_state[4]*1e3,curr_state[5]*1e3,curr_state[6]*1e3]
                jd = base_jd + timestep*i/86400
                centuries = (jd-2451545)/36525
                r_ecef, v_ecef = eci2ecef(r_eci,v_eci,centuries,jd,lod,xp,yp,ddpsi,ddeps)
                lat,lon,alt = ecef2lla(r_ecef[0],r_ecef[1],r_ecef[2])
                
                for visibility in sat["visibilities"]:
                    if visibility[0] == i:
                        sat_pos_and_visibility = [name,lat[0][0],lon[0][0],visibility[2],visibility[3]]
                        sat_visibilities.append(sat_pos_and_visibility)
            with open(base_directory+'sat_visibilities/step'+str(i)+'.csv','w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for vis in sat_visibilities:
                    csvwriter.writerow(vis)
    if not settings["process_obs_only"]:
        if not os.path.exists(base_directory+'overlaps'):
            os.mkdir(base_directory+'overlaps')
        for i in range(len(steps)):
            overlaps = []
            visibilities = []
            for sat in satellites:                
                for visibility in sat["visibilities"]:
                    if visibility[0] == i:
                        visibilities.append((visibility[2],visibility[3]))
            for visibility in visibilities:
                if visibilities.count(visibility) > 1:
                    if visibility not in overlaps:
                        overlaps.append(visibility)
            with open(base_directory+'overlaps/step'+str(i)+'.csv','w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for overlap in overlaps:
                    csvwriter.writerow(overlap)
    if not settings["process_obs_only"]:
        if not os.path.exists(base_directory+'sat_observations'):
            os.mkdir(base_directory+'sat_observations')
        for i in range(len(steps)):
            sat_observations = []
            for sat in satellites:
                name = sat["orbitpy_id"]
                states = sat["states"]
                curr_state = states[i]
                r_eci = [curr_state[1]*1e3,curr_state[2]*1e3,curr_state[3]*1e3]
                v_eci = [curr_state[4]*1e3,curr_state[5]*1e3,curr_state[6]*1e3]
                jd = base_jd + timestep*i/86400
                centuries = (jd-2451545)/36525
                r_ecef, _ = eci2ecef(r_eci,v_eci,centuries,jd,lod,xp,yp,ddpsi,ddeps)
                lat, lon, _ = ecef2lla(r_ecef[0],r_ecef[1],r_ecef[2])
                
                for observation in sat["observations"]:
                    if observation[0] <= i and i <= observation[1]+1:
                        sat_pos_and_observation = [name,lat[0][0],lon[0][0],observation[2],observation[3]]
                        sat_observations.append(sat_pos_and_observation)
            with open(base_directory+'sat_observations/step'+str(i)+'.csv','w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for obs in sat_observations:
                    csvwriter.writerow(obs)

    if not os.path.exists(base_directory+'constellation_past_observations'):
        os.mkdir(base_directory+'constellation_past_observations')

    past_observations = []
    for i in range(len(steps)):
        for sat in satellites:
            name = sat["orbitpy_id"]        
            for observation in sat["observations"]:
                if observation[0] <= i and i <= observation[1]+1:
                    prev_obs = None
                    for past_obs in past_observations:
                        if past_obs[1] == observation[2] and past_obs[2] == observation[3]:
                            prev_obs = past_obs
                    if prev_obs is not None:
                        new_observation = [prev_obs[0]+1,observation[2],observation[3]]
                        past_observations.remove(prev_obs)
                    else:
                        new_observation = [1,observation[2],observation[3]]
                    past_observations.append(new_observation)
        with open(base_directory+'constellation_past_observations/step'+str(i)+'.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for obs in past_observations:
                csvwriter.writerow(obs)
    if not settings["process_obs_only"]:
        if not os.path.exists(base_directory+'ground_swaths'):
            os.mkdir(base_directory+'ground_swaths')

        for i in range(len(steps)):
            ground_swath_points = []
            for sat in satellites:
                name = sat["orbitpy_id"]
                states = sat["states"]
                curr_state = states[i]
                r_eci = [curr_state[1]*1e3,curr_state[2]*1e3,curr_state[3]*1e3]
                v_eci = [curr_state[4]*1e3,curr_state[5]*1e3,curr_state[6]*1e3]
                jd = base_jd + timestep*i/86400
                centuries = (jd-2451545)/36525
                fov_points = [[-settings["along_track_ffor"]/2,-settings["cross_track_ffor"]/2],
                                [-settings["along_track_ffor"]/2,settings["cross_track_ffor"]/2],
                                [settings["along_track_ffor"]/2,settings["cross_track_ffor"]/2],
                                [settings["along_track_ffor"]/2,-settings["cross_track_ffor"]/2]]
                for fov_point in fov_points:
                    ground_point_eci = pitchroll2ecisurface(r_eci,v_eci,fov_point[0],fov_point[1])
                    gp_ecef, _ = eci2ecef(ground_point_eci,[0,0,0],centuries,jd,lod,xp,yp,ddpsi,ddeps)
                    gp_lat,gp_lon,gp_alt = ecef2lla(gp_ecef[0],gp_ecef[1],gp_ecef[2])
                    ground_swath_point = [name,gp_lat[0][0],gp_lon[0][0]]
                    ground_swath_points.append(ground_swath_point)
            with open(base_directory+'ground_swaths/step'+str(i)+'.csv','w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for obs in ground_swath_points:
                    csvwriter.writerow(obs)
    if not settings["process_obs_only"]:
        if not os.path.exists(base_directory+'crosslinks'):
            os.mkdir(base_directory+'crosslinks')

        crosslinks = []
        for f in os.listdir(directory+"comm/"):
            csv_tokens = f.split("_")
            first_sat = csv_tokens[0]
            second_sat = csv_tokens[2][:-4]
            with open(directory+"comm/"+f,newline='') as csv_file:
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

        for i in range(len(steps)):
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
                            jd = base_jd + timestep*i/86400
                            centuries = (jd-2451545)/36525
                            r_ecef, v_ecef = eci2ecef(r_eci,v_eci,centuries,jd,lod,xp,yp,ddpsi,ddeps)
                            lat,lon,alt = ecef2lla(r_ecef[0],r_ecef[1],r_ecef[2])
                            lats.append(lat)
                            lons.append(lon)
                    crosslink_location = [crosslink[2],crosslink[3],lats[0][0][0],lons[0][0][0],lats[1][0][0],lons[1][0][0]]
                    crosslink_locations.append(crosslink_location)
            with open(base_directory+'crosslinks/step'+str(i)+'.csv','w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for obs in crosslink_locations:
                    csvwriter.writerow(obs)
    if not os.path.exists(base_directory+'events'):
        os.mkdir(base_directory+'events')
    if len(settings["event_csvs"]) > 0:
        events = []
        for filename in settings["event_csvs"]:
            with open(filename,newline='') as csv_file:
                csvreader = csv.reader(csv_file, delimiter=',', quotechar='|')
                i = 0
                for row in csvreader:
                    if i < 1:
                        i=i+1
                        continue
                    row = [float(i) for i in row]
                    events.append(row)
        for i in range(len(steps)):            
            events_per_step = []
            step_time = i*settings["step_size"] 
            for event in events:
                if event[2] <= step_time and step_time <= (event[2]+event[3]):
                    event_per_step = [event[0],event[1],event[4]] # lat, lon, start, duration, severity
                    events_per_step.append(event_per_step)
            with open(base_directory+'events/step'+str(i)+'.csv','w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for event in events_per_step:
                    csvwriter.writerow(event)

    print("Processed mission!")

if __name__ == "__main__":
    cross_track_ffor = 60 # deg
    along_track_ffor = 60 # deg
    cross_track_ffov = 0 # deg
    along_track_ffov = 0 # deg
    agility = 1 # deg/s
    num_planes = 5
    num_sats_per_plane = 5
    settings = {
        "directory": "./missions/25_sats_prelim/",
        "step_size": 10,
        "duration": 1,
        "plot_interval": 5,
        "plot_duration": 2/24,
        "plot_location": ".",
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "uniform", # can be "event" or "static"
        "preplanned_observations": None,
        "event_csvs": [],
        "plot_clouds": False,
        "plot_rain": False,
        "plot_obs": True,
        "cross_track_ffor": cross_track_ffor,
        "along_track_ffor": along_track_ffor,
        "cross_track_ffov": cross_track_ffov,
        "along_track_ffov": along_track_ffov,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "planner": "dp",
        "process_obs_only": True
    }

    mission_name = "oa_het_9"
    cross_track_ffor = 90 # deg
    along_track_ffor = 90 # deg
    cross_track_ffov = 1 # deg
    along_track_ffov = 1 # deg
    agility = 0.01 # deg/s
    num_planes = 4
    num_sats_per_plane = 4
    var = 4 # deg lat/lon
    num_points_per_cell = 10
    simulation_step_size = 10 # seconds
    simulation_duration = 1 # days
    event_frequency = 1e-5 # events per second
    event_duration = 21600 # second
    experiment_settings = {
        "event_duration": event_duration,
        "planner": "dp",
        "reobserve_reward": 2,
        "reward": 10
    }
    settings = {
        "directory": "./missions/"+mission_name+"/",
        "step_size": simulation_step_size,
        "duration": simulation_duration,
        "initial_datetime": datetime.datetime(2020,1,1,0,0,0),
        "grid_type": "event", # can be "event" or "static"
        "point_grid": "./coverage_grids/"+mission_name+"/event_locations.csv",
        "preplanned_observations": None,
        "event_csvs": ["./events/"+mission_name+"/events.csv"],
        "cross_track_ffor": cross_track_ffor,
        "along_track_ffor": along_track_ffor,
        "cross_track_ffov": cross_track_ffov,
        "along_track_ffov": along_track_ffov,
        "num_planes": num_planes,
        "num_sats_per_plane": num_sats_per_plane,
        "agility": agility,
        "process_obs_only": False,
        "planner": "dp",
        "reward": 10,
        "reobserve_reward": 2,
        "experiment_settings": experiment_settings
    }
    process_mission(settings)