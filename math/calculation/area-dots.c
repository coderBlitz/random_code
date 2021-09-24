/*	CMSC 455 - hw3 circles part
	TODO: Use OpenCL to calculate grids
	TODO: Figure out OpenCL batching technique (perfect grids, max utilization, etc.)

	Batching notes:
		* Perfect square makes it easy, though it may compute more than necessary
		 and/or waste space.
		* Adaptive square (read rectangles) could help to increase useful area.
		* Max utilization may be more overhead than it's worth

For counting/summing, try (blogspot thing):
    kernel void AtomicSum(global int* sum){
        local int tmpSum[1];
        if(get_local_id(0)==0){
            tmpSum[0]=0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        atomic_add(&tmpSum[0],1);
        barrier(CLK_LOCAL_MEM_FENCE);
        if(get_local_id(0)==(get_local_size(0)-1)){
            atomic_add(sum,tmpSum[0]);
        }
    }
*/

#include<stdio.h>

// Could manually calculate upper bound, and make estimates. PITA to do so.
static inline int circles(double x, double y){
	return ((x*x - 4*(x + y) + y*y + 8) > 1.0) && ((x*x + y*y - 4*y + 4) <= 4.0) && (x*x + y*y <= 9.0);
}

// Sanity check with verifiable quantity (a square).
static inline int square(double x, double y){
	return (x >= -2.0 && x <= 2.0) && (y >= 0.0 && y <= 2.0);
}

// Can probably manually calculate from image (math discord problem)
static inline int more_circles(double x, double y){
	return ((x*x - 10*x + 25 + y*y) <= 25.0) && ((x*x + y*y + 10*y + 25) > 25.0) && ((x*x + y*y + 20*y + 100) <= 100.0);
}

/*	Uses a grid with given interval size. Counts dots in area using the given
	 function, if it returns true.
*/
double area_est(const double interval, int const (*in)(double, double)){
	const double start_x = -3.0;
	const double start_y = -0.125;
	const double end_x = 1.125;
	const double end_y = 3.125;

	unsigned long x_it = 0;
	unsigned long y_it = 0;
	double x = start_x;
	double y;
	register size_t count = 0;
	while(x <= end_x){
		y_it = 0;
		y = start_y;
		while(y <= end_y){
			if(in(x, y)) ++count;

			y = start_y + (++y_it) * interval;
		}
		x = start_x + (++x_it) * interval;
	}
	//printf("Count: %lu\txit: %lu\tyit: %lu\n", count, x_it, y_it);

	double ret = count * (interval * interval);
	return ret;
}

int main(){
	printf("Running the stuff..\n");

	double res;
	double intvl = 1.0;
	for(int i = 1;i <= 14;++i){
		intvl = 1.0 / (1 << i);
		//intvl /= 10.0;
		res = area_est(intvl, circles);
		printf("Int: %le\tRes: %16.8lf\n", intvl, res);
	}
}
