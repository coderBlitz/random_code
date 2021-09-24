/*	Calculates circle intersection points without trig functions
	Triangle formed by both centers, and either of the two intersection points.
	Two sides lengths are circle radii, and last is difference of center
	 positions.
	Use Heron's formula for area of that triangle, then get the triangle height
	 from the area.
	This new side forms a right triangle with both of the radii, so take one
	 to find third length of right triangle.
	Length is distance from center of the circle whose radius was used, so find
	 that point from the center (in direction of other circle center).
	Then `height` distance from this point, at a 90 degree (and -90), will give
	 the points of intersection.
*/

#include<math.h>
#include<stdio.h>

// Custom double absolute value (for curiosity)
static inline double dabs(double f){
	unsigned long val = (0x7FFFFFFFFFFFFFFF & *((unsigned long *)&f));
	return *(double *)&val;
}

struct circle{
	double x;
	double y;
	double r; // Radius (assumed positive)
};

int main(){
	// (0,0,3) and (2,2,1) intersect at (2.707, 1.293) and (1.293, 2.707)
	// (0,0,3) and (sqrt(8), sqrt(8), 1) intersect at
	// (0,-10,10) and (5,0,5) intersect at (0, 0) and (8, -4)
	struct circle a = {0, -10, 10};
	struct circle b = {5, 0, 5};

	double d = sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
	printf("d = %lf\n", d);

	double rp = a.r + b.r;
	double rm = fabs(a.r - b.r);

	if(d > rp || d < rm){
		printf("No intersection.\n");

		return 0;
	}else if(d == 0 && rm == 0){
		printf("Infinite intersections (same circle).\n");

		return 0;
	}else if(d == rp){
		printf("Single point intersection.\n");

		double int_x = a.r * (b.x - a.x) / d; // unit vector scaled
		double int_y = a.r * (b.y - a.y) / d; // unit vector scaled
		printf("(%.12lf, %.12lf)\n", int_x, int_y);
		return 0;
	}

	// Heron's area formula for triangle (half of the trapezoid formed by both points)
	double s = (rp + d) / 2.0; // Semiperimeter of triangle
	//double area = sqrt(s * (s - a.r) * (s - b.r) * (s - d));
	double area = sqrt(d*d*d*d - (rp*rp * rm*rm)) / 4.0; // Reduced version, without `S` (should be equivalent)
	printf("Semi-perim = %lf\nArea = %lf\n", s, area);

	double h = 2 * area / d; // Height of normal. Half the chord length.
	printf("Height = %.12lf\n", h);

	// Since vertical creates a right-triangle, use pythagorean theorem to get third leg.
	double int_pt = sqrt((a.r * a.r) - (h * h));
	printf("Intersection = %.12lf\n", int_pt);

	// Get the point where the chord intersects the difference vector, from circle a
	double int_x = int_pt * (b.x - a.x) / d + a.x; // unit vector scaled
	double int_y = int_pt * (b.y - a.y) / d + a.y; // unit vector scaled
	printf("Chord int: (%.12lf, %.12lf)\n", int_x, int_y);

	// Calculate the circle intersections.
	/*
	Simply the cosine and sine of the difference unit vector, rotated 90 degrees.
	 Then shifted to intersection point, and scaled to h (half the chord).
	cos(theta) = (x1 - x0)/hypot; sin(theta) = (y1-y0)/hypot;
	cos(theta + pi/2) = -sin(theta); sin(theta + pi/2) = cos(theta)
	*/
	double x1 = int_x + h*(b.y - a.y) / d;
	double x2 = int_x - h*(b.y - a.y) / d;
	double y1 = int_y - h*(b.x - a.x) / d;
	double y2 = int_y + h*(b.x - a.x) / d;
	printf("(%22.16lf, %22.16lf)\n(%22.16lf, %22.16lf)\n", x1, y1, x2, y2);
}
