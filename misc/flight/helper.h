#ifndef HELPER_H_
#define HELPER_H_

#include<GL/glew.h>
#include<math.h>

void perspective(float *mat, float fovy, float aspect, float zNear, float zFar){
	for(int i = 0;i < 16;++i) mat[i] = 0;

	float f = 1.0/tan(fovy/2.0);
	mat[0] = f/aspect;
	mat[5] = f;

	float diff = zNear - zFar;
	mat[10] = (zFar + zNear)/diff;
	mat[11] = 2*zFar*zNear/diff;
	mat[14] = -1.0;
}

void lookAt(float *mat,
	GLfloat eyeX,
	GLfloat eyeY,
	GLfloat eyeZ,
	GLfloat centerX,
	GLfloat centerY,
	GLfloat centerZ,
	GLfloat upX,
	GLfloat upY,
	GLfloat upZ){
	for(int i = 0;i < 16;++i) mat[i] = 0;

	float F[] = {centerX - eyeX, centerY - eyeY, centerZ - eyeZ};
	float mag = sqrt(F[0]*F[0] + F[1]*F[1] + F[2]*F[2]);
	mat[11] = -mag; // By observation
	float f[] = {F[0]/mag, F[1]/mag, F[2]/mag};

	float up[] = {upX, upY, upZ};
	mag = sqrt(up[0]*up[0] + up[1]*up[1] + up[2]*up[2]);
	float UP[] = {up[0]/mag, up[1]/mag, up[2]/mag};

	// Cross by un-normalized F by observation
	float s[] = {F[1]*UP[2] - F[2]*UP[1], F[2]*UP[0] - F[0]*UP[2], F[0]*UP[1] - F[1]*UP[0]}; // Cross
	mag = sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
	float S[] = {s[0]/mag, s[1]/mag, s[2]/mag};

	float u[] = {S[1]*f[2] - S[2]*f[1], S[2]*f[0] - S[0]*f[2], S[0]*f[1] - S[1]*f[0]};

	mat[0] = S[0]; // Normalized S by observation
	mat[1] = S[1];
	mat[2] = S[2];

	mat[4] = u[0];
	mat[5] = u[1];
	mat[6] = u[2];

	mat[8] = -f[0];
	mat[9] = -f[1];
	mat[10] = -f[2];

	mat[15] = 1.0;
}

#endif
