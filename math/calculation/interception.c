/*	Calculate interception of two objects in 2d space.
	Note:
	The quadratic I originally used required the direction of the interceptor
	 for the calculation, which defeats the purpose. Instead I used the alternate
	 found on a post about this, which uses one of the other two "Law of cosines"
	 equations. This only requires the interceptor speed, which makes sense.
*/

#include<math.h>
#include<stdio.h>

struct vector{
	double x;
	double y;
	double mag;
};
struct point{
	double x;
	double y;
	struct vector v;
};

void set_v(struct point *p, double x, double y, double vel){
	double mag = sqrt(x*x + y*y);
	if(mag != 0){
		p->v.x = x / mag * vel;
		p->v.y = y / mag * vel;
	}

	p->v.mag = vel;
}

int main(){
	struct point a = {0,0, {1,0,1}};
	struct point b = {3,5, {0,0, 5.0/3.0}}; // Chaser/interceptor
	//set_v(&b, 3, -4, 5.0/3.0);

	double d = sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
	struct point D = {b.x - a.x, b.y - a.y, {}};
	printf("d = %.12lf\n", d);

	double aa = (b.v.mag * b.v.mag) - (a.v.mag * a.v.mag);
	double bb = 2 * (D.x * a.v.x + D.y * a.v.y);
	double cc = -d*d;
	double sum = bb*bb - 4 * aa  * cc;
	printf("Sum = %.12lf\n", sum);

	double t = (sqrt(sum) - bb)/(2*aa);
	printf("t = %.12lf\n", t);
	struct point pi = {a.x + a.v.x * t, a.y + a.v.y * t, a.v};
	printf("pi = (%.12lf, %.12lf)\n", pi.x, pi.y);

	struct point vc = {b.x, b.y, {(pi.x - b.x)/t, (pi.y - b.y)/t, b.v.mag}};
	printf("vc = (%.12lf, %.12lf, %.12lf)\n", vc.v.x, vc.v.y, vc.v.mag);
	//struct point vi = {b.x + b.v.x * t, b.y + b.v.y * t, b.v};
	//printf("vi = (%.12lf, %.12lf)\n", vi.x, vi.y);
}
