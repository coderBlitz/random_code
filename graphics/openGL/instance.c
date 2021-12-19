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
	-1.0, -1.0,
	-1.0, 1.0,
	1.0, 1.0,
	1.0, 1.0,
	1.0, -1.0,
	-1.0, -1.0
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
	// Initialize window
	if(!glfwInit()){
		fprintf(stderr, "GLFW init failed.\n");
		return -1;
	}

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
	size_t offset_size = 2 * total * sizeof(*offset);
	offset = malloc(offset_size);
	if(offset == NULL){
		fprintf(stderr, "Can't alloc offsets.\n");
		return -1;
	}

	/*	Calculate offsets
	*/
	for(int i = 0;i < row_count;i += 1){
		for(int j = 0;j < col_count;j += 1){
			int idx = 2 * (col_count * i + j);
			offset[idx] = (((4.0 * j + 1.0) * col_size) - 1.0) / col_size; // Offset from left edge
			offset[idx+1] = (((4.0 * i + 1.0) * row_size) - 1.0) / row_size; // Offset from bottom edge
		}
	}

	/*for(int i = 0;i < 2*total;i += 2){
		printf("(%.2f, %.2f)\n", offset[i], offset[i+1]);
		for(int j = 0;j < 12;j += 2){
			printf("  (%.2f, %.2f)\n", offset[i] + square_vertex_data[j]*col_size, offset[i+1] + square_vertex_data[j+1]*row_size);
		}
	}*/

	// Initialize GL buffers
	GLuint square_vertex_buffer;
	glGenBuffers(1, &square_vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, square_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(square_vertex_data), square_vertex_data, GL_STATIC_DRAW);
	glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribBinding(0, 0);
	glBindVertexBuffer(0, square_vertex_buffer, 0, 2 * sizeof(GLfloat));
	glVertexBindingDivisor(0, 0);

	GLuint square_offset_buffer;
	glGenBuffers(1, &square_offset_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, square_offset_buffer);
	glBufferData(GL_ARRAY_BUFFER, offset_size, offset, GL_DYNAMIC_DRAW);
	glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribBinding(1, 1);
	glBindVertexBuffer(1, square_offset_buffer, 0, 2 * sizeof(GLfloat));
	glVertexBindingDivisor(1, 1);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	/*	Initialize shaders
	*/
	GLuint programID = LoadShaders("instance-vshdr.glsl", "instance-fshdr.glsl");
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

		glDrawArraysInstanced(GL_TRIANGLES, 0, 2*3, total); // Draw 'total' squares, with 6 vertices each, starting index 0
		//glDrawArrays(GL_TRIANGLES, 0, 2*3);

		/* per-loop stuff
		*/
		if(glGetError() != GL_NO_ERROR){
			fprintf(stderr, "Error in main loop\n");
			break;
		}

		// Changes in col dimensions
		if(glfwGetKey(window, GLFW_KEY_RIGHT)){
			col_count += 1;

			total = row_count * col_count;
			size_tmp = 2 * total * sizeof(*offset);

			// Realloc if needed
			if(size_tmp > offset_size){
				printf("Reallocating col..\n");

				offset_size = 2*size_tmp;
				GLfloat *temp = realloc(offset, offset_size);
				if(temp == NULL){
					fprintf(stderr, "Realloc failed.\n");
					break;
				}
				offset = temp;
			}

			goto col_update;
		}else if(glfwGetKey(window, GLFW_KEY_LEFT) && col_count > 1){
			col_count -= 1;

			total = row_count * col_count;
		col_update:
			col_size = 1.0 / (2 * col_count - 1); // Square width
			scale[0] = col_size;
			printf("col_count = %d\tcol_size = %.2f\n", col_count, col_size);

			for(int i = 0;i < row_count;i += 1){
				for(int j = 0;j < col_count;j += 1){
					int idx = 2 * (col_count * i + j);
					offset[idx] = (((4.0 * j + 1.0) * col_size) - 1.0) / col_size; // Offset from left edge
					offset[idx+1] = (((4.0 * i + 1.0) * row_size) - 1.0) / row_size; // Offset from bottom edge
				}
			}

			goto buffer_update;
		}else if(glfwGetKey(window, GLFW_KEY_UP)){
			row_count += 1;

			total = row_count * col_count;
			size_tmp = 2 * total * sizeof(*offset);

			// Realloc if needed
			if(size_tmp > offset_size){
				printf("Reallocating row..\n");

				offset_size = 2*size_tmp;
				GLfloat *temp = realloc(offset, offset_size);
				if(temp == NULL){
					fprintf(stderr, "Realloc failed.\n");
					break;
				}
				offset = temp;
			}

			goto row_update;
		}else if(glfwGetKey(window, GLFW_KEY_DOWN) && row_count > 1){
			row_count -= 1;

			total = row_count * col_count;
		row_update:
			row_size = 1.0 / (2 * row_count - 1); // Square height
			scale[5] = row_size;
			printf("row_count = %d\trow_size = %.2f\n", row_count, row_size);

			for(int i = 0;i < row_count;i += 1){
				for(int j = 0;j < col_count;j += 1){
					int idx = 2 * (col_count * i + j);
					offset[idx] = (((4.0 * j + 1.0) * col_size) - 1.0) / col_size; // Offset from left edge
					offset[idx+1] = (((4.0 * i + 1.0) * row_size) - 1.0) / row_size; // Offset from bottom edge
				}
			}

			// Update GL buffer
		buffer_update:
			glUniformMatrix4fv(uniform_model, 1, GL_FALSE, scale);

			glBindBuffer(GL_ARRAY_BUFFER, square_offset_buffer);
			glBufferData(GL_ARRAY_BUFFER, offset_size, NULL, GL_DYNAMIC_DRAW);
			glBufferSubData(GL_ARRAY_BUFFER, 0, offset_size, offset);
		}

		// Control framerate
		/*frame++;
		if(current.tv_sec - previous_frame.tv_sec >= 1){
			fps = frame;
			frame = 0;
			previous_frame = current;
			printf("FPS: %u\n", fps);
		}*/

		glfwPollEvents();
		glfwSwapBuffers(window);
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window));

	// Cleanup
	glfwDestroyWindow(window);
	glfwMakeContextCurrent(NULL);
	glfwTerminate();

	free(offset);

	glDeleteBuffers(1, &square_vertex_buffer);
	glDeleteBuffers(1, &square_offset_buffer);

	return 0;
}
