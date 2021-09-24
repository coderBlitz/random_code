import sys
import math
import time

print "Ideal Projectile Simulation v1.0"


START_Y_POS = 10.0 # Height at which object is launched from
INIT_VELOCITY = 15.0 # Velocity in generic measurement (distance unit/second)
INIT_ANGLE = 20.0 # Launch angle in degrees

x_pos=0.0
y_pos = START_Y_POS # Will be current stats of projectile
X_VELOCITY = INIT_VELOCITY * math.cos(INIT_ANGLE * math.pi/180.0) # Get horizontal speed
Y_VELOCITY = INIT_VELOCITY * math.sin(INIT_ANGLE * math.pi/180.0) # Get vertical speed

sys.stdout.write("INITIAL HEIGHT: %.03f\n"%START_Y_POS)
sys.stdout.write("INITIAL SPEED: %.03f\n"%INIT_VELOCITY)
sys.stdout.write("ANGLE: %.03f degrees\n"%INIT_ANGLE)
sys.stdout.write("LAUNCH!\n")

cur_time = 0.0
start = time.time() # Get launch/start time
current = start
while(y_pos > 0.0): # While projectile is still in flight
	current = time.time()
	cur_time = current-start # Get difference

	x_pos = cur_time * X_VELOCITY
	y_pos = START_Y_POS + Y_VELOCITY * cur_time + (0.5 * -9.8 * cur_time * cur_time) # Get height
	# y = init_vert_velocity * time + 0.5 * vert.accel * time^2
	# vert.accel AKA Gravity
	# Gravity is -9.8 m/s/s

	sys.stdout.write("\rTime: %.05f\t X_POS: %.06f\tY_POS: %.06f"%(cur_time,x_pos,y_pos))

current = time.time()
cur_time = current - start
sys.stdout.write("\nLANDED\nProjectile went %.04f distance units in %.06f seconds.\n"%(x_pos,cur_time))

