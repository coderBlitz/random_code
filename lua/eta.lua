--[[
Numerically compute speed to maintain a fixed ETA, for a given starting speed
 and distance.

NOTE: Numerically, speed appears to have closed form INIT_SPEED * exp(-x / ETA), where x is distance remaining.
Not certain if exact, but it certainly appears to be big theta of this (asymptotically identical).

Closed form is correct. Can be represented as the differential equation:
(D - y) / y' = E, y(0) = 0

Sol:
	y = D - D exp(- x / ETA) = D - D exp(- x * S_0 / D)
	y' = S_0 exp(- x * S_0 / D)
--]]

-- Parameters
DT = 0.005
MAX_ITERS = 4000

INIT_SPEED = 15
INIT_DISTANCE = 100

-- Main stuff
speed = INIT_SPEED
distance = INIT_DISTANCE

eta = distance / speed

test = distance
for it = 0,MAX_ITERS do
	time = it * DT
	print(string.format("Time = %6.3f\tSpeed = %.17f\tDistance = %5.17f\td = %.17f", time, speed, distance, test))

	test = test - test * (DT / eta) -- test is similar to distance, out to around 13 decimal places
	distance = distance - speed * DT
	speed = distance / eta
end
