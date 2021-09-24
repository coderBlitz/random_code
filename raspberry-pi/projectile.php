<?php
printf("Ideal Projectile Simulation v1.0\n");


$START_Y_POS = 10.0;// Height at which object is launched from
$INIT_VELOCITY = 15.0;// Velocity in generic measurement (distance unit/second)
$INIT_ANGLE = 20.0;// Launch angle in degrees

$x_pos=0.0; $y_pos = $START_Y_POS;// Will be current stats of projectile
$X_VELOCITY = $INIT_VELOCITY * cos($INIT_ANGLE * M_PI/180.0);// Get horizontal speed
$Y_VELOCITY = $INIT_VELOCITY * sin($INIT_ANGLE * M_PI/180.0);// Get vertical speed

printf("INITIAL HEIGHT: %.03f\n",$START_Y_POS);
printf("INITIAL SPEED: %.03f\n",$INIT_VELOCITY);
printf("ANGLE: %.03f degrees\n",$INIT_ANGLE);
printf("LAUNCH!\n");

$cur_time = 0.0;
$start = microtime(true);// Get launch/start time
echo "START: ".$start."\n";
$current = $start;
while($y_pos > 0.0){// While projectile is still in flight
	$current = microtime(true);
	$cur_time = $current-$start;// Get difference

	$x_pos = $cur_time * $X_VELOCITY;
	$y_pos = $START_Y_POS + $Y_VELOCITY * $cur_time +(0.5 * -9.8 * $cur_time * $cur_time);// Get height
	// y = init_vert_velocity * time + 0.5 * vert.accel * time^2
	// vert.accel AKA Gravity
	// Gravity is -9.8 m/s/s

	printf("\rTime: %.05f\t X_POS: %.06f\tY_POS: %.06f",$cur_time,$x_pos,$y_pos);
}
$current = microtime(true);
$cur_time = $current - $start;
printf("\nLANDED\nProjectile went %.04f distance units in %.06f seconds.\n",$x_pos,$cur_time);
?>
