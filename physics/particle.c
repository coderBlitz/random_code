#include"particle.h"
#include<stdint.h>
#include<math.h>

const uint64_t K = 9000000000;

float netForce(struct _particle A, struct _particle B){
	float dx = A.x - B.x;
	float dy = A.y - B.y;
	float dz = A.z - B.z;
	float distance = dx*dx + dy*dy + dz*dz;

	return K * A.q*B.q / distance;
}

float fieldStrength(struct _particle A, float distance){
	particle q;
	q.x = A.x + distance;
	q.y = A.y;
	q.z = A.z;
	q.q = 1.0;

	return netForce(A, q);
}
