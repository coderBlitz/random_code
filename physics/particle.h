#ifndef PARTICLE_H
#define PARTICLE_H

typedef struct _particle{
// Position in meters
	float x;
	float y;
	float z;
// Charge in Coulombs
	float q;
} particle;

// The net force (in Newtons) between 2 particles in a vector from A to B
float netForce(struct _particle, struct _particle);

// The strength (in Coulombs) of the field generated by particle A
float fieldStrength(struct _particle, float);

#endif
