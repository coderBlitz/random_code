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

const uint16_t width = 1000;
const uint16_t height = width;
const char *title = "Instance experiment";

const GLfloat square_vertex_data[] = {
	-1.0, -1.0, 0.0,
	-1.0, 1.0, 0.0,
	1.0, 1.0, 0.0,
	1.0, 1.0, 0.0,
	1.0, -1.0, 0.0,
	-1.0, -1.0, 0.0
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

/**	Copy vertices to arr, with offset added for each dimension (fixed 2 in this case)
**/
static inline void copy_with_offset2(GLfloat *arr, GLfloat *const offset, GLfloat const *verts, size_t N){
	for(size_t i = 0;i < N;i++){
		arr[2*i + 0] = verts[2*i + 0] + offset[0];
		arr[2*i + 1] = verts[2*i + 1] + offset[1];
		printf("\t%g = %g + %g\n", arr[2*i + 0], verts[2*i + 0], offset[0]);
		printf("\t%g = %g + %g\n", arr[2*i + 1], verts[2*i + 1], offset[1]);
	}
}

int main(){
	glewExperimental = GL_TRUE;
	// Initialize window
	if(!glfwInit()){
		fprintf(stderr, "GLFW init failed.\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
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

	/*	Initialize offset data/matrices
	*/
	uint32_t row_count = 10; // How many squares per row
	uint32_t col_count = 10;
	uint32_t total = row_count * col_count;
	float row_size = 1.0 / (2 * row_count - 1); // Square height
	float col_size = 1.0 / (2 * col_count - 1); // Square width

	printf("%d %.2f x %.2f squares\n", total , 2*col_size, 2*row_size);
	scale[0] = col_size;
	scale[5] = row_size;

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
	color[0] = 0.3;
	color[1] = 1.0;
	color[2] = 0.5;

	color[3] = 0.6;
	color[4] = 0.6;
	color[5] = 0.6;

	color[6] = 1.0;
	color[7] = 0.5;
	color[8] = 0.3;

	color[9] = 0.6;
	color[10] = 0.6;
	color[11] = 0.6;

	color[12] = 0.5;
	color[13] = 0.3;
	color[14] = 1.0;

	color[15] = 0.6;
	color[16] = 0.6;
	color[17] = 0.6;

	/* Calculate offsets
	*/
	for(int i = 0;i < row_count;i += 1){
		for(int j = 0;j < col_count;j += 1){
			long base_idx = col_count * i + j;
			long off_idx = 3 * base_idx; // 3-dimensional coordinates

			offset[off_idx] = (((4.0 * j + 1.0) * col_size) - 1.0) / col_size; // Offset from left edge
			offset[off_idx + 1] = (((4.0 * i + 1.0) * row_size) - 1.0) / row_size; // Offset from bottom edge
			offset[off_idx + 2] = 0.0;
		}
	}

	/*for(int i = 0;i < row_count;i += 1){
		for(int j = 0;j < col_count;j += 1){
			long base_idx = col_count * i + j;
			long off_idx = 12 * base_idx; // 6 vertices per square coordinates

			printf("Square %ld:\n", base_idx);
			printf("  (%g, %g)\n", verts[off_idx + 0], verts[off_idx + 1]);
			printf("  (%g, %g)\n", verts[off_idx + 2], verts[off_idx + 3]);
			printf("  (%g, %g)\n", verts[off_idx + 4], verts[off_idx + 5]);
			printf("  (%g, %g)\n", verts[off_idx + 6], verts[off_idx + 7]);
			printf("  (%g, %g)\n", verts[off_idx + 8], verts[off_idx + 9]);
			printf("  (%g, %g)\n", verts[off_idx + 10], verts[off_idx + 11]);
		}
	}*/

	/* Initialize GL buffers
	Notes/observations:
		When divisor for color_buffer is 0, the color attribute increments (for stride)
		 for every vertex. So this is a per-vertex setting.
		When divisor for color_buffer is 1, the color attribute is incremented (for stride)
		 for every 1 instance (every `count` primitive draws, from drawArraysInstanced call).
	*/
	GLuint square_vertex_buffer;
	glGenBuffers(1, &square_vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, square_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(square_vertex_data), square_vertex_data, GL_STATIC_DRAW); // Instanced vertices
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(0, 0);

	GLuint square_offset_buffer;
	glGenBuffers(1, &square_offset_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, square_offset_buffer);
	glBufferData(GL_ARRAY_BUFFER, offset_size, offset, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(1, 1); // Divisor of 1 increments s_o_b by 3 (size) every 1 instance. Divisor 2 every 2, etc.

	GLuint color_buffer;
	glGenBuffers(1, &color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
	glBufferData(GL_ARRAY_BUFFER, color_size, color, GL_STATIC_DRAW);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(2, 0);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

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
		//glEnableVertexAttribArray(0);
		//glEnableVertexAttribArray(1);

		glDrawArraysInstanced(GL_TRIANGLES, 0, 2*3, total); // Draw 'total' squares, each 6 vertices (2D*3 sets), starting index 0
		//glDrawArrays(GL_TRIANGLES, 0, 2*3);

		//glDisableVertexAttribArray(1);
		//glDisableVertexAttribArray(0);

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
