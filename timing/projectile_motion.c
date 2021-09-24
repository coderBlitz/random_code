#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<unistd.h>
#include<sys/time.h>

// Will simulate position of generic projectile on earth
/* CURRENT PROGRAM
   The "simulation" is the launch of a generic projectile with no wind resistance,
   changing gravity based on altitude above surface, and curvature of earth.
   ** Achieving orbit is possible **
*/

int main(int argc,char **argv){
	printf("Ideal Projectile Simulation v1.0\n");

	struct timeval tval_start,tval_current,tval_diff;// Will be used for timing

	float START_Y_POS;// Height at which object is launched from
	float INIT_VELOCITY;// Velocity in generic measurement (distance unit/second)
	float INIT_ANGLE;// Launch angle in degrees

	for(int i=0;i<argc;i++){// See if 'i' or '-i' is given to allow manual entry
		if(argv[i][0] == 'i' || argv[i][1] == 'i'){
			printf("Enter starting height (meters): ");
			scanf("%f",&START_Y_POS);
			printf("Enter Initial Velocity (m/s): ");
			scanf("%f",&INIT_VELOCITY);
			printf("Enter Firing angle: ");
			scanf("%f",&INIT_ANGLE);
			break;
		}else{
			START_Y_POS = 10;// Height at which object is launched from
			INIT_VELOCITY = 15;// Velocity in generic measurement (distance unit/second)
			INIT_ANGLE = 20;// Launch angle in degrees
		}
	}
	float x_pos=0, y_pos = START_Y_POS;// Will be current stats of projectile in metres
	float X_VELOCITY = INIT_VELOCITY * cos(INIT_ANGLE * M_PI/180.0);// Get horizontal speed
	float Y_VELOCITY = INIT_VELOCITY * sin(INIT_ANGLE * M_PI/180.0);// Get vertical speed
	float gravity = -9.8/pow((y_pos+6400)/6400,2);// Get gravity at different altitudes above surface
	long surface = 0;// Will be how far down the surface of the earth is, changes based on distance

	printf("INITIAL HEIGHT: %.03fm\n",START_Y_POS);
	printf("INITIAL SPEED: %.03fm/s\n",INIT_VELOCITY);
	printf("ANGLE: %.03f degrees\n",INIT_ANGLE);
	printf("LAUNCH!\n");

	float cur_time;
	gettimeofday(&tval_start,NULL);// Get launch/start time
	while(y_pos >= surface){// While projectile is still in flight
		gettimeofday(&tval_current,NULL);
		timersub(&tval_current,&tval_start,&tval_diff);// Get difference
		cur_time = (tval_diff.tv_sec + (tval_diff.tv_usec / 1e6))*10;// Make time easier for calculations

		// Environment parameters
		gravity = -9.8*pow((6371.0/(6371.0+((y_pos-surface)/1000.0))),2);// Update current gravitational pull
		surface = (x_pos/1609.34)*-0.125946930732;// Curve of earth decreases ~7.98 in/mi or ~12.6 cm/km

		// Location parameters
		x_pos = cur_time * X_VELOCITY;
		y_pos = START_Y_POS + Y_VELOCITY * cur_time + (0.5 * gravity * cur_time * cur_time) - surface;
		// y = init_vert_velocity * time + 0.5 * vert.accel * time^2
		// vert.accel AKA Gravity
		// Gravity is -9.8 m/s/s at sea level

		printf("\rTime: %5.04fs\t  X_POS: %11.5fm\t Y_POS: %8.06fm",cur_time,x_pos,y_pos);
		printf("\tSurface: %ld",surface);
		fflush(stdout);
if(y_pos<0) printf("\n\nGravity: %5.06f\t  X_POS: %11.5fm\t Y_POS: %8.06fm\n\n",gravity,x_pos,y_pos);
	}
	gettimeofday(&tval_current,NULL);
	timersub(&tval_current,&tval_start,&tval_diff);
	cur_time = tval_diff.tv_sec + (tval_diff.tv_usec / 1e6);

	printf("\nLANDED\nProjectile went %.04f distance units in %.06f seconds.\n",x_pos,cur_time);
}
