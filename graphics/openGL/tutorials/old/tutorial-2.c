#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<unistd.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include<glm/gtc/matrix_transform.hpp>
#include"common/shader.hpp"

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
		// Projection matrix: 45 degree Field of View (FOV), aspect ratio, display range: 0.1 <-> 100 units
		glm::mat4 Projection = glm::perspective(glm::radians(45.0f), (float) WIDTH/ (float) HEIGHT, 0.1f, 100.0f);

		glm::mat4 View = glm::lookAt(
			glm::vec3(4,3,3),	// Camera at (4,3,3)
			glm::vec3(0,0,0),	// Look at origin (0,0,0)
			glm::vec3(0,1,0)	// Up vector, in Y+ direction
		);
		glm::mat4 Model = glm::mat4(1.0f);
		glm::mat4 mvp = Projection * View * Model;

		GLuint MatrixID = glGetUniformLocation(programID, "MVP");

		glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

		glUseProgram(programID);
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);

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

		glfwSwapBuffers(window);
		glfwPollEvents();
	}while(	glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
				glfwWindowShouldClose(window) == 0);

	glfwTerminate();
	return 0;
}
