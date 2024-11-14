import sys
num = input("Number to convert to binary: ")

binary = []
while num != 0:
	binary.append(num % 2)
	num /= 2

for i in reversed(range(len(binary))):
	sys.stdout.write("%d"%binary[i])
print
