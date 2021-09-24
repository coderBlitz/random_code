def factorial(num):
	sum = 1
	for i in range(1,num+1):
		sum *= i
	return sum

n = input("Enter number to find factorial: ")
total = factorial(n)
print "The factorial of %d is %d"%(n,total)
