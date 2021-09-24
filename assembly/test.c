#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<time.h>
#include<unistd.h>

extern float add(float,float);

int main(){
	srand(time(0));

	float A = 5.5;
	float B = 4.359;
	float C = 0.0f;

	C = add(A, B);
	printf("%0.5f + %0.5f = %0.5f\n", A, B, C);
}
