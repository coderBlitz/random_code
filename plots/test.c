#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<time.h>

int main(){
	srandom(time(0));
	float A = 10.0;
	int N = 1000;

	FILE *plot = popen("gnuplot -p","w");
	fprintf(plot, "set title 'C plot window'\n");

	fprintf(plot,"plot '-' pt 7\n");
	float X = 0.0, Y=0.0;
	for(int i=0;i<N;i++){
		X = (float)random()/(float)RAND_MAX * A;
		Y = (float)random()/(float)RAND_MAX * A;
		fprintf(plot, "%f %f\n", X, Y);
	}
	fprintf(plot, "e");

	fclose(plot);
}
