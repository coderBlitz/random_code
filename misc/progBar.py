import time
import sys

for x in range(101):
	sys.stdout.write("\r|")
	for y in range(0,101,2):
		if y<x:
			sys.stdout.write('-')
			sys.stdout.flush()
			continue
		if y==100:
			sys.stdout.write('|')
			sys.stdout.flush()
			continue
		if y>=x:
			sys.stdout.write(' ')
			sys.stdout.flush()
			continue
	print"%3d%%"%x,
	time.sleep(.1)
print
