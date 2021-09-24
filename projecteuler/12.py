# What is the value of the first triangle number to have over five hundred divisors?

# Note: Every triangular number is divisible by 3, or is equivalent to 1 (mod 9)

# 6: 4 divisors
# 28: 6 divisors
# 60: 12 divisors
# 600: 24 divisors

# Triangular number = n*(n+1)/2
# T(7) = 28
# 7*8/2 => 7 (2*2)
# Factors of n and (n+1), except one 2

# Use number theory prime factorization to calculate number of divisors
# 2^166 * 3^2 == 841824943102600080885322463644579019321817144754176 -- smallest number with 501 divisors
