import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.figure(figsize=(12, 6))
ax = plt.axes()
offset = 1
plt.scatter(10,100,s=100)
ax.annotate("Sentinel-2", xy=(10-offset,100-offset), ha='right', va='top',fontsize=12)
plt.scatter(30,10,s=100)
ax.annotate("SBG", xy=(30-offset,10-offset), ha='right', va='top',fontsize=12)
#plt.scatter(1,10)
ellipse = Ellipse(xy=(10,10),width=10,height=10,edgecolor='r',fc='None',lw=2)
ax.add_patch(ellipse)
ax.annotate("New mission?", xy=(10,10), ha='center', va='center',fontsize=12)
plt.xlim([0,50])
plt.ylim([0,120])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Spatial resolution (m)",fontsize=12)
plt.ylabel("Spectral resolution (nm)",fontsize=12)
plt.savefig("satellite_tradespace.png",dpi=300)