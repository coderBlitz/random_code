import thread
import time

def thing( name):
	print "Thread '%s' starting..."%name
	for x in range(6):
		print "%d"%x
		time.sleep(1)

print "Starting Threads..."
try:
	thread.start_new_thread(thing,("first",))
	thread.start_new_thread(thing,("second",))
except:
	print "thread creation failed..."

while 1:
	pass
print "Done"
