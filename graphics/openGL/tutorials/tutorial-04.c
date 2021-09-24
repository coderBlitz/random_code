#include<math.h>
#include<stdint.h>
#include<stdio.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include"shader.c"

const uint16_t width = 1024;
const uint16_t height = 768;

const GLfloat cube_color_data[] = {
	0.0, 0.0, 0.0,
	0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, // end 1
	0.0, 0.0, 0.0,
	0.0, 0.0, 0.0,
	0.0, 0.0, 0.0, // end 2
	0.0, 0.0, 1.0,
	0.0, 0.0, 1.0,
	0.0, 0.0, 1.0, // end 3
	0.0, 0.0, 1.0,
	0.0, 0.0, 1.0,
	0.0, 0.0, 1.0, // end 4
	0.0, 1.0, 0.0,
	0.0, 1.0, 0.0,
	0.0, 1.0, 0.0, // end 5
	0.0, 1.0, 0.0,
	0.0, 1.0, 0.0,
	0.0, 1.0, 0.0, // end 6
	0.0, 1.0, 1.0,
	0.0, 1.0, 1.0,
	0.0, 1.0, 1.0, // end 7
	0.0, 1.0, 1.0,
	0.0, 1.0, 1.0,
	0.0, 1.0, 1.0, // end 8
	1.0, 0.0, 0.0,
	1.0, 0.0, 0.0,
	1.0, 0.0, 0.0, // end 9
	1.0, 0.0, 0.0,
	1.0, 0.0, 0.0,
	1.0, 0.0, 0.0, // end 10
	1.0, 0.0, 1.0,
	1.0, 0.0, 1.0,
	1.0, 0.0, 1.0, // end 11
	1.0, 0.0, 1.0,
	1.0, 0.0, 1.0,
	1.0, 0.0, 1.0, // end 12
};

const GLfloat cube_vertex_data[] = {
	-1.0, -1.0, -1.0,
	1.0, -1.0, -1.0,
	-1.0, 1.0, -1.0, // end 1 (back)
	-1.0, 1.0, -1.0,
	1.0, 1.0, -1.0,
	1.0, -1.0, -1.0, // end 2
	1.0, -1.0, -1.0,
	1.0, 1.0, -1.0,
	1.0, -1.0, 1.0, // end 3 (right)
	1.0, -1.0, 1.0,
	1.0, 1.0, 1.0,
	1.0, 1.0, -1.0, // end 4
	1.0, 1.0, -1.0,
	-1.0, 1.0, -1.0,
	1.0, 1.0, 1.0, // end 5 (top)
	1.0, 1.0, 1.0,
	-1.0, 1.0, 1.0,
	-1.0, 1.0, -1.0, // end 6
	-1.0, 1.0, -1.0,
	-1.0, -1.0, -1.0,
	-1.0, 1.0, 1.0, // end 7 (left)
	-1.0, 1.0, 1.0,
	-1.0, -1.0, 1.0,
	-1.0, -1.0, -1.0, // end 8
	-1.0, -1.0, -1.0,
	1.0, -1.0, -1.0,
	1.0, -1.0, 1.0, // end 9 (bottom)
	1.0, -1.0, 1.0,
	-1.0, -1.0, -1.0,
	-1.0, -1.0, 1.0, // end 10
	-1.0, -1.0, 1.0,
	1.0, -1.0, 1.0,
	-1.0, 1.0, 1.0, // end 11 (front)
	-1.0, 1.0, 1.0,
	1.0, 1.0, 1.0,
	1.0, -1.0, 1.0 // end 12
};

const GLfloat triangle_vertex_data[] = {
	-1.0f, -1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	0.0f, 1.0f, 0.0f
};

const GLfloat triangle_color_data[] = {
	1.0, 0.0, 0.0,
	0.0, 1.0, 0.0,
	0.0, 0.0, 1.0
};

GLfloat shift[] = {
	1.0f, 0.0f, 0.0f, 3.0f,
	0.0f, 1.0f, 0.0f, 1.0f,
	0.0f, 0.0f, 1.0f, 3.0f,
	0.0f, 0.0f, 0.0f, 1.0f
};
GLfloat scale[] = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
};
GLfloat rotation[] = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
};
GLfloat projection[] = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
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

	GLFWwindow *window = glfwCreateWindow(width, height, "Tutorial 4", NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Window failed to open\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;
	//glfwSwapInterval(0); // Effectively disables V-sync (60 fps cap)

	if(glewInit() != GLEW_OK){
		fprintf(stderr, "Glew failed\n");
		return -1;
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);


	// GL stuff
	GLuint VertexArrayID; // These 3 are basically a minimum required
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);


	// Prepare the buffers
	GLuint cubeVertexBuffer;
	glGenBuffers(1, &cubeVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, cubeVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertex_data), cube_vertex_data, GL_STATIC_DRAW);

	GLuint cubeColorBuffer;
	glGenBuffers(1, &cubeColorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, cubeColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_color_data), cube_color_data, GL_STATIC_DRAW);

	GLuint triangleVertexBuffer;
	glGenBuffers(1, &triangleVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, triangleVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangle_vertex_data), triangle_vertex_data, GL_STATIC_DRAW);

	GLuint triangleColorBuffer;
	glGenBuffers(1, &triangleColorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, triangleColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangle_color_data), triangle_color_data, GL_STATIC_DRAW);


	// Prepare for loop
	GLuint programID = LoadShaders("vshader04.glsl", "fshader04.glsl");
	printf("ProgramID: %u\n", programID);
	if(programID == 0){
		fprintf(stderr, "Program failed to load correctly.\n");
		return -3;
	}

	//lookAt(rotation, 4,3,3, 0,0,0, 0,1,0);
	perspective(projection, M_PI/4.0, (float)4.0/3.0, 0.1f, 100.0f);

	//for(int i = 0;i < 16;++i) printf("%f%c", rotation[(i/4)+(i%4)*4], ((i&3) == 3)?'\n':' ');
	//printf("\n");

	GLint uniform_WindowSize = glGetUniformLocation(programID, "WindowSize"); // Used to pass windows size
	GLuint uniform_model = glGetUniformLocation(programID, "model");
	GLuint uniform_view = glGetUniformLocation(programID, "view");
	GLuint uniform_projection = glGetUniformLocation(programID, "projection");


	register int count = 0;
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	do{
		glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		lookAt(rotation, 6*cos(count*M_PI/180.0),2,6*sin(count*M_PI/180.0), 0,0,0, 0,1,0);

		glUseProgram(programID);

		// Pass vars to shaders
		glUniform2f(uniform_WindowSize, width, height);
		glUniformMatrix4fv(uniform_model, 1, GL_FALSE, scale);
		glUniformMatrix4fv(uniform_view, 1, GL_TRUE, rotation);
		glUniformMatrix4fv(uniform_projection, 1, GL_TRUE, projection);

		/***** DRAW STUFF *****/
		glEnableVertexAttribArray(0); // Start cube
		glBindBuffer(GL_ARRAY_BUFFER, cubeVertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glEnableVertexAttribArray(1); // Colors
		glBindBuffer(GL_ARRAY_BUFFER, cubeColorBuffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glDrawArrays(GL_TRIANGLES, 0, 12*3);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0); // End cube


		glUniformMatrix4fv(uniform_model, 1, GL_TRUE, shift); // Shift matrix for triangle

		/*glEnableVertexAttribArray(0); // Start cube
		glBindBuffer(GL_ARRAY_BUFFER, cubeVertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glEnableVertexAttribArray(1); // Colors
		glBindBuffer(GL_ARRAY_BUFFER, cubeColorBuffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glDrawArrays(GL_TRIANGLES, 0, 12*3);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0); // End cube*/

		glEnableVertexAttribArray(0); // Start triangle
		glBindBuffer(GL_ARRAY_BUFFER, triangleVertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glEnableVertexAttribArray(1); // Colors
		glBindBuffer(GL_ARRAY_BUFFER, triangleColorBuffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glDrawArrays(GL_TRIANGLES, 0, 3);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0);
		/***** END DRAW STUFF *****/

		if(glGetError() != GL_NO_ERROR){
			fprintf(stderr, "Error in main loop\n");
			break;
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
		++count;
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	glfwTerminate();

	return 0;
}
