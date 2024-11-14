def triangle(array, level, N):
	if N < 1:
		print("Invalid arguments")

	array.append([1])

	i = 1
	while(i < level):
		a = array[level-1][i]
		b = array[level-1][i-1]
		array[level].append(a+b)
		i += 1

	if(level != 0):
		array[level].append(1)

	if((level+1) == N):
		return
	else:
		triangle(array, level+1, N)

N = 10;
print("N = %d"%N)

array = []
triangle(array,0,N)

for row in array:
	for num in row:
		print(num,end=" ")
	print()
