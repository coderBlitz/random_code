from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import math


lats = [38.27452,38.885974,36.142591,]
lons = [-76.413161,-77.020919,-86.776552,]
labels = ['PAX River','Washington, DC','Nashville, TN',]

avgx = sum(lons)/float(len(lons))
avgy = sum(lats)/float(len(lats))

# Latitude 1 deg = 110.574 km
# Longitude 1 deg = 111.320*cos(latitude) km
height = (max(lats)-min(lats)) * 110.574
width = (max(lons)-min(lons)) * 111.320 * math.cos(math.radians(avgy))
height *= 1000
width *= 1000

print("Center at %.6f,%.6f"%(avgx,avgy))
print("Width: %.4f    Height: %.4f"%(width,height))

map = Basemap(resolution='h',
			  projection='ortho',
			  lat_0=avgy,
			  lon_0=avgx,
			  llcrnrx=-width,
			  llcrnry=-height,
			  urcrnrx=width,
			  urcrnry=height)
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral',lake_color='aqua')

x,y = map(lons,lats)

map.drawcoastlines()

for xpt,ypt in zip(x,y):
  map.plot(xpt,ypt,marker='D',color='m')

for xpt,ypt,label in zip(x,y,labels):
  lonpt,latpt = map(xpt,ypt,inverse=True)
  plt.text(xpt+10000,ypt+10000,label,color='indigo')

plt.savefig('test.png')
plt.show()
