#ifndef SHADERS_C_
#define SHADERS_C_

#include<stdio.h>
#include<stdlib.h>
#include<error.h>
#include<GL/glew.h>

#define MAX_FILE_SIZE 0x1000

GLuint LoadShaders(const char *vertex_path, const char *fragment_path){
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Get vertex shader source
	char *vertex_source = (char *)malloc(MAX_FILE_SIZE * sizeof(*vertex_source));
	if(vertex_source == NULL){
		fprintf(stderr, "Could not allocate vertex source\n");
		return 0;
	}

	FILE *fp = fopen(vertex_path, "rb");
	if(fp == NULL){
		perror("Vertex file cannot be opened");
		return 0;
	}

	size_t vertex_source_size = fread(vertex_source, sizeof(*vertex_source), MAX_FILE_SIZE, fp);
	printf("Vertex source size: %lu\n", vertex_source_size);
	fclose(fp);

	//printf("Vertex source:\n%s\n", vertex_source);

	// Get fragment shader source
	char *fragment_source = (char *)malloc(MAX_FILE_SIZE * sizeof(*fragment_source));
	if(fragment_source == NULL){
		fprintf(stderr, "Could not allocate fragment source\n");
		return 0;
	}

	fp = fopen(fragment_path, "rb");
	if(fp == NULL){
		perror("Fragment path cannot be opened");
		return 0;
	}

	size_t fragment_source_size = fread(fragment_source, sizeof(*fragment_source), MAX_FILE_SIZE, fp);
	printf("Fragment source size: %lu\n", fragment_source_size);
	fclose(fp);

	//printf("Fragment source:\n%s\n", fragment_source);


	GLint res = GL_FALSE;
	int infoLogLength = 0;

	// Compiling vertex
	printf("Compiling vertex shader\n");
	char const * test = vertex_source;
	glShaderSource(VertexShaderID, 1, &test, NULL);
	glCompileShader(VertexShaderID);

	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &res);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
	if(res == GL_FALSE){
		printf("Program failed with result: %d\n", res);
	}
	if(infoLogLength > 0){
		char msg[infoLogLength+1];
		glGetShaderInfoLog(VertexShaderID, infoLogLength, NULL, msg);
		printf("'%s'\n", msg);
	}

	// Compile fragment
	printf("Compiling fragment shader\n");
	char const *test2 = fragment_source;
	glShaderSource(FragmentShaderID, 1, &test2, NULL);
	glCompileShader(FragmentShaderID);

	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &res);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
	if(res == GL_FALSE){
		printf("Program failed with result: %d\n", res);
	}
	if(infoLogLength > 0){
		char msg[infoLogLength+1];
		glGetShaderInfoLog(FragmentShaderID, infoLogLength, NULL, msg);
		printf("'%s'\n", msg);
	}

	// Link program
	printf("Linking..\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	glGetProgramiv(ProgramID, GL_LINK_STATUS, &res);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &infoLogLength);
	if(res == GL_FALSE){
		printf("Program failed with result: %d\n", res);
	}
	if(infoLogLength > 0){
		char msg[infoLogLength+1];
		glGetProgramInfoLog(ProgramID, infoLogLength, NULL, msg);
		printf("'%s'\n", msg);
	}

	// Cleanup
	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);


	free(vertex_source);
	free(fragment_source);

	return ProgramID;
}

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
