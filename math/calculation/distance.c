/* Description: Calculates distance between two latitude/longitude/altitude positions.
				Converts "spherical" lat/lon to cartesian, use vector calculus to get
				 angle between, then just arc length to find distance.
*/

#include<math.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>

#define TORAD (M_PI/180.0)
#define SEMIMAJOR 6378137.0
#define SEMIMAJOR2 40680631590769.0
#define SEMIMINOR 6356752.314245
#define SEMIMINOR2 40408299984659.163229

struct vector{
	double X;
	double Y;
	double Z;
	double mag;
};

struct sphere_coord{
	double R;
	double T;
	double P;
};

int main(int argc, char *argv[]){
	struct vector u = {0.0, 0.0, 0.0, 0.0};
	struct vector v = {0.0, 0.0, 0.0, 0.0};

	struct sphere_coord pos1 = {0.0, -76.421621, 38.276898};
	struct sphere_coord pos2 = {0.0, -76.390536, 38.298094};

	pos1.T *= TORAD;
	pos1.P *= TORAD;
	pos2.T *= TORAD;
	pos2.P *= TORAD;

	double phi = sin(M_PI/2.0 - pos1.P);
	u.X = cos(pos1.T) * phi;
	u.Y = sin(pos1.T) * phi;
	u.Z = cos(M_PI/2.0 - pos1.P);

	phi = sin(M_PI/2.0 - pos2.P);
	v.X = cos(pos2.T) * phi;
	v.Y = sin(pos2.T) * phi;
	v.Z = cos(M_PI/2.0 - pos2.P);

	printf("(%.5lf, %.5lf, %.5lf) -> <%.5lf, %.5lf, %.5lf>\n", pos1.R, pos1.T, pos1.P, u.X, u.Y, u.Z);
	printf("(%.5lf, %.5lf, %.5lf) -> <%.5lf, %.5lf, %.5lf>\n", pos2.R, pos2.T, pos2.P, v.X, v.Y, v.Z);

	double dot = u.X*v.X + u.Y*v.Y + u.Z*v.Z;
	printf("Dot: %.9lf\n", dot);

	double angle = acos(dot);
	printf("angle: %.9lf (%.4lf deg)\n", angle, angle / TORAD);

	double depth = sqrt(SEMIMAJOR2*cos(pos1.P)*cos(pos1.P) + SEMIMINOR2*sin(pos1.P)*sin(pos1.P)) + pos1.R;
	printf("Sea level radius: %lf\n", depth);

	double distance = angle*depth;
	printf("Distance: %.4lf\n", distance);

	return 0;
}
