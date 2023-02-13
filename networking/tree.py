import threading
from time import sleep

count_lock = threading.Lock()
list_lock = threading.Lock()
count = 0
threads = []
N = 10

"""	Tree send data to each thread, to maximize communication bandwidth.
	Demonstration for potential IPC or other network(-like) communications.
"""
def tree(num):
	global count
	global threads
	global count_lock
	global list_lock

	#print("Thread", num, "counted!")
	count_lock.acquire()
	count += 1
	count_lock.release()

	if 2*num > N:
		return
		
	#print(num, "launching thread", 2*num)
	thread = threading.Thread(target=tree, args=(2*num,))
	thread.start()
	
	list_lock.acquire()
	threads.append(thread)
	list_lock.release()

	res = bin(num).count("1")
	if res != 1:
		# Power of 2 case
		#print("Not power of 2 thread with num", num)
		num *= 2 # Go left
	
	num = 2*num + 1 # Next right proc
	while num <= N:
		#print(num//2, "launching thread", num)
		thread = threading.Thread(target=tree, args=(num,))
		thread.start()
		
		list_lock.acquire()
		threads.append(thread)
		list_lock.release()
		
		num = 2*num + 1

print("Calling tree with count =", count)
tree(1)
print("Waiting for threads..")
for t in threads:
	t.join()
print("After tree with count =", count)
