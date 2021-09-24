#include<stdio.h>
#include<stdlib.h>
#include<math.h>

float deg(float angle){
	return angle * (float)180.0 / (float)M_PI;
}

int main(int argc, char *argv[]){
	if(argc < 6){
		printf("Usage: %s {x1} {y1} {z1} {x2} {y2} {z2}\n", argv[0]);
		exit(1);
	}

	int v1[3], v2[3];
	v1[0] = atoi(argv[1]);
	v1[1] = atoi(argv[2]);
	v1[2] = atoi(argv[3]);

	v2[0] = atoi(argv[4]);
	v2[1] = atoi(argv[5]);
	v2[2] = atoi(argv[6]);

	int dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];

	int v1_mag_2 = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
	float v1_mag = sqrt(v1_mag_2);
	int v2_mag_2 = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
	float v2_mag = sqrt(v2_mag_2);

	printf("V1 Direction angles: {%.4f, %.4f, %.4f}\n", deg(acos(v1[0]/v1_mag)), deg(acos(v1[1]/v1_mag)), deg(acos(v1[2]/v1_mag)));
	printf("V2 Direction angles: {%.4f, %.4f, %.4f}\n", deg(acos(v2[0]/v2_mag)), deg(acos(v2[1]/v2_mag)), deg(acos(v2[2]/v2_mag)));

	printf("Dot-Product: %d\n", dot_product);

	printf("V1 Mag: %.4f (sqrt %d)\n", v1_mag, v1_mag_2);
	printf("V2 Mag: %.4f (sqrt %d)\n", v2_mag, v2_mag_2);

	printf("Angle: %.4f degrees\n", deg(acos((float)dot_product / (float)(v1_mag*v2_mag))) );
}
