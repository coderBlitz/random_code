def triangle(array, level, N):
	if N < 1:
		print("Invalid target of %d"%N)
	array.append([1])

	i = 1;
	while i < (level-1):
		a = array[level-2][i]
		b = array[level-2][i-1]
		array[level-1].append(a+b)

		i += 1
	# If not the first row
	if(level != 1):
		array[level-1].append(1)

	if(level == N):
		return
	else:
		return triangle(array, level+1, N)

N = 10
print("Test with N = %d"%N)
array = []
triangle(array, 1, N)

for row in array:
	for i in row:
		print(i, end=' ')
	print()
