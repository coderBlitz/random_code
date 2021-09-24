/* Description: Logic used to give an programmed entity movement closer to that
				 of an actual aircraft, specifically when loitering.
				Though parts could be adapted for other manners of flight.
*/

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<time.h>

#define TORAD (M_PI/180.0)

struct position_t{
	double X;
	double Y;
	double Z;
};

int main(){
	srand(time(0));
/* Note:	The turn directions here are compass-based, so positive turn rate
			 is indicative of a clockwise turn, and vice-versa.
*/

	unsigned long delay_usec = 250000;
	double delay_fraction = delay_usec / 1e6;

	struct position_t center = {0.0, 0.0, 0.0};
	struct position_t ent = {2500.0, 2500.0, 0.0};

	float bearing = 0.0; // Bearing 0 deg == North

	struct position_t dir = {ent.X - center.X, ent.Y - center.Y, ent.Z - center.Z};
	struct position_t ent_vector = {sin(bearing*TORAD), cos(bearing*TORAD), 0.0};
	struct position_t ent_math_vector = {cos(bearing*TORAD), -sin(bearing*TORAD), 0.0};

	double mag = 10.0; // Degrees per second (positive is clockwise)
	double db = 0.0; // Change in bearing (independent var)
	double dd = 0.0; // Change in distance (dependent var)
	double velocity = 150.0; // Velocity M/S

	double point_velocity = 15.0; // Center target attributes
	double point_mag = 5.0;
	double point_db = 0.0;
	double point_bearing = 90.0;

	double cross = 0.0;
	double lr_cross = 0.0;
	double distance = 0.0; // Distance from center
	double prev_distance = 0.0;
	double diff = 0.0;

	double radius_min = (180.0/mag)*velocity/M_PI; // (time to turn 360 degrees)*speed = 2*pi*radius
	double radiusM = 15.0;
	if(radiusM < radius_min) radiusM = radius_min;
	printf("Radius min: %.4lf\tRadius target: %.4lf\n", radius_min, radiusM);

	unsigned long count = 0;
	unsigned long N = 300/delay_fraction;
	while(++count <= N){
		dir.X = ent.X - center.X;
		dir.Y = ent.Y - center.Y;
		dir.Z = ent.Z - center.Z;

		ent_vector.X = sin(bearing*TORAD);
		ent_vector.Y = cos(bearing*TORAD);
		ent_math_vector.X = cos(bearing*TORAD);
		ent_math_vector.Y = -sin(bearing*TORAD);

		ent.X += ent_vector.X * velocity * delay_fraction;
		ent.Y += ent_vector.Y * velocity * delay_fraction;

		cross = dir.X*ent_math_vector.Y - dir.Y*ent_math_vector.X;
		lr_cross = dir.X*ent_vector.Y - dir.Y*ent_vector.X;

		prev_distance = distance;
		distance = sqrt(dir.X*dir.X + dir.Y*dir.Y); // Distance from center
		dd = distance - prev_distance;
		diff = distance - radiusM;

		printf("%8.4lf,%8.4lf,%8.4lf,%8.4lf,%8.4lf\n", ent.X, ent.Y, distance, center.X, center.Y);

		// Determines clockwise or counter-clockwise
		if(cross > 0) db = mag;
		else db = -mag;

		// If smallest radius possible is inside target radius
		if(radiusM > 2*radius_min){
			if(distance < (radiusM - radius_min)) db = 0.0;
		}

		//////// This logic should work, but it has not been tested ////////
		// If greatly outside radius, turn towards center
		// If outside radius and increasing distance, turn inward
		// If outside radius, elminate turns toward outside
		// If inside radius, but near, reduce turn-in rate
		// If inside radius, but near and decreasing distance, turn outward

		if(diff > radius_min){
			if(lr_cross > 0){
				db = -mag;
			}else if(lr_cross < 0){
				db = mag;
			}
		}
		else if(diff > 0){
			if(dd > -velocity/mag/2){
				if(lr_cross > 0)
					db = -mag;
				else
					db = mag;
			}else{ // If decreasing distance (below threshold of if-statement)
				if(lr_cross > 0 && cross > 0){ // If turning outward
					//db *= 0.8; // How much to adjust turn out rate when heading toward radius
				}else if(lr_cross < 0 && cross < 0){
					//db *= 0.8;
				}
			}
		}else if(diff > -velocity){
			if(lr_cross > 0){
				if(dd < 0)
					db = mag;
				else if(db < 0)
					db *= 0.2; // Rate at which to turn inward, when close to radius
			}else if(lr_cross < 0){
				if(dd < 0)
					db = -mag;
				else if(db > 0)
					db *= 0.2;
			}
		}

		//printf("time: %.2lf\tdist: %.4lf\tdb: %.2lf\tdd: %lf\n", count*delay_usec/1e6, distance, db, dd, lr_cross);
		bearing += db*delay_fraction;

		point_db = 2.0 * point_mag * (float)rand()/RAND_MAX - point_mag; // Should give random direction within range
		point_bearing += point_db;

		center.X += point_velocity * cos(point_bearing*TORAD) * delay_fraction;
		center.Y += point_velocity * sin(point_bearing*TORAD) * delay_fraction;

		//usleep(delay_usec);
	}

	return 0;
}
