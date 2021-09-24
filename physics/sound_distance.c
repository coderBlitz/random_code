#include<stdio.h>
#include<math.h>

// Useful to estimate distance to an event which you can see and hear sounds from
// fireworks are what "inspired" this. Just get delay of sound from visual event

int main(){
	printf("Distance Estimate using sound delay\n");

	double delay;
	printf("Time Delay of sound from visual cue (ms): ");
	scanf("%lf",&delay);

	double T = 20;// Degrees Celcius
	double C;// Speed of sound (m/s) at temperature T
	C = 331.3*sqrt(1 + T/273.15);

	double D = C*(delay/1000);

	printf("Estimated distance at %.2lf degrees C:\n%.4lf m\n%.4lf ft\n",T,D,D*3.37);
}
