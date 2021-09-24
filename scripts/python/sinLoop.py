import math

for d in range(91):
	s=0.0
	r=d*math.pi/180
	s=math.sin(r)
	print "Sin(%d)=%f"%(d,s)
