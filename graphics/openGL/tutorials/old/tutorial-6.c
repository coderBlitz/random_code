#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<unistd.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include<glm/gtc/matrix_transform.hpp>
#include"common/shader.hpp"
#include"common/controls.hpp"

static const GLfloat g_vertex_buffer_data[] = {
	-1.0f, -1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	0.0f, 1.0f, 0.0f
};


GLFWwindow* init(const uint16_t W, const uint16_t H, const char *title){
	if(!glfwInit()){
		fprintf(stderr, "GLFW init failed\n");
		exit(-1);
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window;

	window = glfwCreateWindow(W, H, title, NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Window creation failed\n");
		glfwTerminate();
		exit(-1);
	}

	glfwMakeContextCurrent(window);

	glewExperimental = 1;
	if(glewInit() != GLEW_OK){
		fprintf(stderr, "GLEW init failed\n");
		glfwTerminate();
		exit(-1);
	}

	return window;
}
int main(){
	const uint16_t WIDTH = 1600;
	const uint16_t HEIGHT = 900;
	const char *title = "Thingy";

	GLFWwindow* window = init(WIDTH, HEIGHT, title);


	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData( GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	GLuint programID = LoadShaders( "SimpleVertexShader.glsl", "SimpleFragmentShader.glsl" );


	do{
		computeMatricesFromInputs(window);

		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();
		glm::mat4 Model = glm::mat4(1.0f);
		glm::mat4 MVP = ProjectionMatrix * ViewMatrix * Model;

		GLuint MatrixID = glGetUniformLocation(programID, "MVP");

		// Draw things
		glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

		glUseProgram(programID);
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			0,						// Attribute
			3,						// Size (num vertices presumably)
			GL_FLOAT,			// Type
			GL_FALSE,			// Normalized
			0,						// stride
			(void*)0			// array buffer offset
		);
		glDrawArrays(GL_TRIANGLES, 0, 3); // Starting from vertex 0; 3 vertices total -> 1 triangle
		glDisableVertexAttribArray(0);

		// Finish draw
		glfwSwapBuffers(window);
		glfwPollEvents();
	}while(	glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
				glfwWindowShouldClose(window) == 0);

	glfwTerminate();
	return 0;
}
