# Find the last ten digits of the series, 1^1 + 2^2 + 3^3 + ... + 1000^1000.

# Use addition modulus 10^10 (if slow)
# PS: It's not needed, brute approach works easily

N = 1000
nums = [n**n for n in range(1,N+1)]

total = sum(nums)
digits = []
for i in range(10):
	digits.append(total % 10)
	total //= 10
digits.reverse()
print("".join([str(a) for a in digits]))
