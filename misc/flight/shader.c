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

#endif
