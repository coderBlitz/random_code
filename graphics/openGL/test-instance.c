/*	Experiment with instanced rendering. That is, rendering many instances of
	 a single set of vertices.
*/

#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include"shader.c"

const uint16_t width = 500;
const uint16_t height = width;
const char *title = "Instance experiment";

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
	// Initialize window
	if(!glfwInit()){
		fprintf(stderr, "GLFW init failed.\n");
		return -1;
	}

	// TODO: Upgrade to 4.3/4.4
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow *window = glfwCreateWindow(width, height, title, NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Create window failed.\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
	//glfwSwapInterval(0);

	if(glewInit() != GLEW_OK){
		fprintf(stderr, "Glew failed.\n");
		glfwTerminate();
		return -1;
	}

	// Just do this
	GLuint VAO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	const uint32_t total = 2;

	GLfloat *triangles;
	size_t triangles_size = total * 9 * sizeof(*triangles);
	triangles = malloc(triangles_size);
	if(triangles == NULL){
		return -1;
	}

	triangles[0] = -1.0;
	triangles[1] = -1.0;
	triangles[2] = 0.0;

	triangles[3] = 1.0;
	triangles[4] = -1.0;
	triangles[5] = 0.0;

	triangles[6] = 0.0;
	triangles[7] = 1.0;
	triangles[8] = 0.0;

	triangles[9] = 1.0;
	triangles[10] = 1.0;
	triangles[11] = 0.0;

	triangles[12] = -1.0;
	triangles[13] = 1.0;
	triangles[14] = 0.0;

	triangles[15] = 0.0;
	triangles[16] = -1.0;
	triangles[17] = 0.0;

	GLfloat *offset;
	size_t offset_size = total * 3 * sizeof(*offset);
	offset = malloc(offset_size);
	if(offset == NULL){
		fprintf(stderr, "Can't alloc offsets.\n");
		return -1;
	}

	GLfloat *color;
	size_t color_size = 6 * 3 * sizeof(*color); // Num vertices * 3 colors each
	color = malloc(color_size);
	if(color == NULL){
		fprintf(stderr, "Can't alloc colors.\n");
		return -1;
	}
	color[0] = 1.0;
	color[1] = 0.0;
	color[2] = 0.0;

	color[3] = 0.0;
	color[4] = 1.0;
	color[5] = 0.0;

	color[6] = 0.0;
	color[7] = 0.0;
	color[8] = 1.0;

	color[9] = 0.0;
	color[10] = 0.0;
	color[11] = 1.0;

	color[12] = 0.0;
	color[13] = 1.0;
	color[14] = 0.0;

	color[15] = 1.0;
	color[16] = 0.0;
	color[17] = 0.0;

	/* Calculate offsets
	*/
	offset[0] = -0.05;
	offset[1] = 0.0;
	offset[2] = 0.0;

	offset[3] = 0.05;
	offset[4] = 0.0;
	offset[5] = 0.0;

	/* Initialize GL buffers
	*/
	GLuint square_vertex_buffer;
	glGenBuffers(1, &square_vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, square_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, triangles_size, triangles, GL_STATIC_DRAW); // Load data to gl buffer
	glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0); // Set format for shader attribute index (0)
	glVertexAttribBinding(0, 10); // Bind shader attribute index (0) to backing buffer index (10)
	// Bind gl buffer to buffer index. Stride no longer automatic for 0
	glBindVertexBuffer(10, square_vertex_buffer, 0, 3*sizeof(GLfloat));
	glVertexBindingDivisor(10, 0); // Set divisor for buffer index

	GLuint square_offset_buffer;
	glGenBuffers(1, &square_offset_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, square_offset_buffer);
	glBufferData(GL_ARRAY_BUFFER, offset_size, offset, GL_STATIC_DRAW);
	glVertexAttribFormat(1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribBinding(1, 11);
	glBindVertexBuffer(11, square_offset_buffer, 0, 3 * sizeof(GLfloat));
	glVertexBindingDivisor(11, 1);

	GLuint color_buffer;
	glGenBuffers(1, &color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
	glBufferData(GL_ARRAY_BUFFER, color_size, color, GL_STATIC_DRAW);
	glVertexAttribFormat(2, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribBinding(2, 12);
	glBindVertexBuffer(12, color_buffer, 0, 3 * sizeof(GLfloat));
	glVertexBindingDivisor(12, 1);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	/*	Initialize shaders
	*/
	GLuint programID = LoadShaders("color-vshdr.glsl", "color-fshdr.glsl");
	if(!programID){
		fprintf(stderr, "Shaders failed to load\n");
		glfwTerminate();
		return -1;
	}
	glUseProgram(programID);

	/*	Initialize view/perspective and misc uniform data
	No need for view/perspective for default
	*/

	GLint uniform_WindowSize = glGetUniformLocation(programID, "WindowSize"); // Used to pass windows size
	GLuint uniform_model = glGetUniformLocation(programID, "model");
	GLuint uniform_view = glGetUniformLocation(programID, "view");
	GLuint uniform_projection = glGetUniformLocation(programID, "projection");

	glUniform2f(uniform_WindowSize, width, height);
	glUniformMatrix4fv(uniform_model, 1, GL_FALSE, scale);
	glUniformMatrix4fv(uniform_view, 1, GL_TRUE, rotation);
	glUniformMatrix4fv(uniform_projection, 1, GL_TRUE, projection);

	/*	Main loop
	*/
	size_t size_tmp;

	// Frame stuff
	struct timespec current, previous, previous_frame;
	uint16_t fps = 0;
	register uint64_t frame = 0;
	clock_gettime(CLOCK_REALTIME, &current);
	previous = current;
	previous_frame = current;

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	do{
		previous = current;
		clock_gettime(CLOCK_REALTIME, &current);

		glClearColor(0, 0.2, 0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		/*	Draw calls
		*/

		glDrawArraysInstanced(GL_TRIANGLES, 0, 3, total); // Draw 'total' squares, each 6 vertices (2D*3 sets), starting index 0
		//glDrawArrays(GL_TRIANGLES, 0, 2*3);

		/* per-loop stuff
		*/
		if(glGetError() != GL_NO_ERROR){
			fprintf(stderr, "Error in main loop\n");
			break;
		}

		// Control framerate
		frame++;
		if(current.tv_sec - previous_frame.tv_sec >= 1){
			fps = frame;
			frame = 0;
			previous_frame = current;
			printf("FPS: %u\n", fps);
		}

		glfwPollEvents();
		glfwSwapBuffers(window);
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window));

	// Cleanup
	glfwDestroyWindow(window);
	glfwMakeContextCurrent(NULL);
	glfwTerminate();

	free(offset);
	free(color);

	glDeleteBuffers(1, &square_vertex_buffer);
	glDeleteBuffers(1, &square_offset_buffer);
	glDeleteBuffers(1, &color_buffer);

	return 0;
}
