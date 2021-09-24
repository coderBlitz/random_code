import random
import time
import numpy
import sys

N = 1000000
argc = len(sys.argv)
if argc == 2:
	N = int(sys.argv[1])

print("N:", N)
#A = [random.getrandbits(32) for i in range(N)]
A = numpy.random.randint(2**32, size=N)

#begin = time.process_time_ns()
begin = time.perf_counter()
A.sort(kind="quicksort")
#end = time.process_time_ns()
end = time.perf_counter()

diff = (end-begin) #/ (10**9)
print(diff, "seconds")
