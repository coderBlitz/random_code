# What is the sum of the digits of 2^1000

num = 2 ** 1000

count = 0
while num > 0:
	count += num % 10
	num //= 10

print(num)
print(count)
