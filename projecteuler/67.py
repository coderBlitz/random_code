# Find the maximum total from top to bottom of the triangle below: (see file)

#	Bottom-up approach to find maximum path possible from a given node, based on
#	 the largest path possible through either child.

nums = []
with open("67.txt","r") as fp:
	for line in fp:
		raw = line.split(",")
		nums.append([int(n) for n in raw])

#print(nums)

res = nums.copy()
# Starting at second-to-last row, find max path
N = len(nums) - 2

while N >= 0:
	# For each entry in the row
	for i,num in enumerate(nums[N], 0):
		# Get the highest possible path starting with this entry, by choosing
		#  largest child.
		best = max(res[N+1][i], res[N+1][i+1]) + num
		res[N][i] = best
		#print(num, best)

	N -= 1

#print(res)
print("Max:", res[0][0])
