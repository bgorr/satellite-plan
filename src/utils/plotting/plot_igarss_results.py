import matplotlib.pyplot as plt
import numpy as np


two_meas_types = [1597,1844,1967]
dp = [1597,818]
rdp = [1844,980]
hrdp = [1967,1058]
three_meas_types = [818,980,1058]
  
n=2
r = np.arange(n) 
width = 0.25
  

plt.bar(r - width, dp, color = 'r', 
        width = width, edgecolor = 'black', 
        label='DP') 
plt.bar(r, rdp, color = 'b', 
        width = width, edgecolor = 'black', 
        label='RDP') 
plt.bar(r + width, hrdp, color = 'g', 
        width = width, edgecolor = 'black', 
        label='HRDP') 
  
plt.xlabel("Number of measurement types required",fontsize=12) 
plt.ylabel("Number of fully observed events",fontsize=12) 

# plt.grid(linestyle='--') 
plt.xticks(r,['Two','Three'],fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12) 
  
plt.show()


dp = [33.9,38.0,53.5]
rdp = [48.0,46.4,61.8]
hrdp = [50.7,52.4,66.5]
  
n=3
r = np.arange(n) 
width = 0.25
  

plt.bar(r - width, dp, color = 'r', 
        width = width, edgecolor = 'black', 
        label='DP') 
plt.bar(r, rdp, color = 'b', 
        width = width, edgecolor = 'black', 
        label='RDP') 
plt.bar(r + width, hrdp, color = 'g', 
        width = width, edgecolor = 'black', 
        label='HRDP') 
  
plt.xlabel("Event duration",fontsize=12) 
plt.ylabel("Percent of events that are fully observed",fontsize=12) 

# plt.grid(linestyle='--') 
plt.xticks(r,['2 hours','4 hours','6 hours'],fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12) 
  
plt.show() 