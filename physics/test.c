#include"particle.h"

#include<stdio.h>
#include<stdint.h>

int main(int argc, char *argv[]){
	particle a, b;
	a.x = 0;
	a.y = 0;
	a.z = 0;
	a.q = 0.000003; // 7 micro-Coulombs
	b.x = 0.086;
	b.y = b.z = 0;
	b.q = -0.0000055; // -5.5 micro-Coulombs

	float force = netForce(a,b);
	printf("Force: %.6f N\n", force);

	force = fieldStrength(a, 1.0);
	printf("Field: %.6f N/C\n", force);
}
