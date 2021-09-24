#include<stdint.h>
#include<stdio.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include"shader.c"

const uint16_t width = 1024;
const uint16_t height = 768;

const GLfloat vertex_buffer_data[] = {
	-1.0f, -1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	0.0f, 1.0f, 0.0f
};

int main(){
	glewExperimental = GL_TRUE;
	if(!glfwInit()){
		fprintf(stderr, "GLFW Failed\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow *window = glfwCreateWindow(width, height, "Totorial 2", NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Window failed to open\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;

	if(glewInit() != GLEW_OK){
		fprintf(stderr, "Glew failed\n");
		return -1;
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);


	// GL stuff
	GLuint VertexArrayID; // These 3 are basically a minimum required
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);


	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data), vertex_buffer_data, GL_STATIC_DRAW);


	GLuint programID = LoadShaders("vshader.glsl", "fshader.glsl");
	printf("ProgramID: %u\n", programID);

	GLint uniform_WindowSize = glGetUniformLocation(programID, "WindowSize"); // Pass window size to frag shader
	do{
		glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(programID);
		glUniform2f(uniform_WindowSize, width, height);
		/***** DRAW STUFF *****/
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glDrawArrays(GL_TRIANGLES, 0, 3);
		glDisableVertexAttribArray(0);
		/***** END DRAW STUFF *****/

		glfwSwapBuffers(window);
		glfwPollEvents();
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	return 0;
}
