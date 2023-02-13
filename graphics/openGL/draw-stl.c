/*	Try to draw an STL file

*/
#include<stdio.h>
#include<stdlib.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include<time.h>

#include"shader.c"
#include<stl.h> // Custom code to read STL files

unsigned int width = 500;
unsigned int height = 500;

float model[] = {
	0.2, 0.0, 0.0, 0.0,
	0.0, 0.2, 0.0, 0.0,
	0.0, 0.0, 0.2, 0.0,
	0.0, 0.0, 0.0, 1.0
};
float view[] = {
	1.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.0, 0.0, 0.0, 1.0
};
float projection[] = {
	1.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.0, 0.0, 0.0, 1.0
};
float base[] = {
	0, 0, 0,
	0, 0, 0,
	0, 0, 0
};

float randf(){
	return (float) rand() / (float)RAND_MAX;
}

float *flatten_stl_triangles(struct triangle *tris, const unsigned int count){
	float *ret = malloc(9 * count * sizeof(*ret));
	if(ret == NULL) return NULL;

	for(unsigned int i = 0;i < count;i++){
		ret[9*i + 0] = tris[i].verts[0].x;
		ret[9*i + 1] = tris[i].verts[0].y;
		ret[9*i + 2] = tris[i].verts[0].z;

		ret[9*i + 3] = tris[i].verts[1].x;
		ret[9*i + 4] = tris[i].verts[1].y;
		ret[9*i + 5] = tris[i].verts[1].z;

		ret[9*i + 6] = tris[i].verts[2].x;
		ret[9*i + 7] = tris[i].verts[2].y;
		ret[9*i + 8] = tris[i].verts[2].z;
	}

	return ret;
}

int main(){
	srand(time(0));
	// Pick file
	const int num_files = 1;
	char *filename = "/home/chris/Documents/freeCAD/box-thing.stl";

	/*	Window setup
	*/
	printf("Init GLFW\n");
	glewExperimental = GL_TRUE;
	if(!glfwInit()){
		fprintf(stderr, "GLFW init failed.\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	printf("Creating window..\n");
	GLFWwindow *window = glfwCreateWindow(width, height, filename, NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Create window failed.\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);

	/*	GL setup
	*/
	if(glewInit() != GLEW_OK){
		fprintf(stderr, "Glew failed.\n");
		glfwTerminate();
		return -1;
	}

	/*	Attempt to make expandable VAO usage
	*/
	GLuint *VAOs = malloc(num_files * sizeof(*VAOs));
	if(VAOs == NULL){
		fprintf(stderr, "VAO alloc failed\n");
		glfwTerminate();
		return -1;
	}
	glGenVertexArrays(num_files, VAOs);
	glBindVertexArray(VAOs[0]);

	/*	Get STL data
	*/
	printf("Fetching data..\n");
	size_t sizes[num_files];
	unsigned int num_triangles[num_files];
	float *triangles[num_files];
	struct triangle *tris;
	for(int i = 0;i < num_files;i++){
		tris = stl_read_triangles_file(filename, num_triangles + i);
		if(tris == NULL){
			fprintf(stderr, "Read error.\n");
			glfwTerminate();
			return -2;
		}
		triangles[i] = flatten_stl_triangles(tris, num_triangles[i]);
		if(triangles[i] == NULL){
			fprintf(stderr, "Flatten error.\n");
			glfwTerminate();
			return -2;
		}
		sizes[i] = 9 * num_triangles[i] * sizeof(**triangles);
		printf("Got %d triangles\n", num_triangles[i]);

		free(tris);
	}

	//num_triangles[0] = 2;
	//sizes[0] = 9 * num_triangles[0] * sizeof(**triangles);

	// Fill the color buffer
	float *colors;
	size_t colors_size = 9 * num_triangles[0] * sizeof(*colors);
	colors = malloc(colors_size);
	if(colors == NULL){
		fprintf(stderr, "Colors alloc failed\n");
		glfwTerminate();
		return -2;
	}
	for(int i = 0;i < 9*num_triangles[0];++i){
		if((i % 4) == 0) colors[i] = 1.0;
		else colors[i] = 0.0;
	}

	// Initialize GL buffers
	/*/ Ideally can use instanced rendering to save memory by reusing same color for
	 every triangle. However, triangles need 3 vertices to draw, but instancing will
	 only use 1 vertex at a time, incrementing according to the divisor (but still
	 only 1 vertex at a time!!!), hence nothing gets drawn.
	In other words, a non-zero divisor makes each instance only use one `size` (set
	 by vertexAttribFormat) worth of attribute data.
	Hence, instancing will not be possible. If desired, using a uniform would suffice.

	See color-instance.c for further notes
	//*/
	GLuint triangle_buffer;
	glGenBuffers(1, &triangle_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, triangle_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizes[0], triangles[0], GL_STATIC_DRAW);
	glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribBinding(0, 10);
	glBindVertexBuffer(10, triangle_buffer, 0, 3 * sizeof(GLfloat));

	GLuint color_buffer;
	glGenBuffers(1, &color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
	glBufferData(GL_ARRAY_BUFFER, colors_size, colors, GL_STATIC_DRAW);
	glVertexAttribFormat(1, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribBinding(1, 11);
	glBindVertexBuffer(11, color_buffer, 0, 3 * sizeof(GLfloat));

	/*GLuint base_buffer;
	glGenBuffers(1, &base_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, base_buffer);
	glBufferData(GL_ARRAY_BUFFER, 9 * sizeof(*base), base, GL_STATIC_DRAW);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(2, 0);*/

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	//glEnableVertexAttribArray(2);

	/*	Initialize shaders
	*/
	GLuint programID = LoadShaders("simple-vshader.glsl", "simple-fshader.glsl");
	if(!programID){
		fprintf(stderr, "Shaders failed to load\n");
		glfwTerminate();
		return -1;
	}
	glUseProgram(programID);

	/*	Initialize view/perspective and misc uniform data
	No need for view/perspective for default
	*/

	//GLint uniform_WindowSize = glGetUniformLocation(programID, "WindowSize"); // Used to pass windows size
	GLuint uniform_model = glGetUniformLocation(programID, "model");
	GLuint uniform_view = glGetUniformLocation(programID, "view");
	GLuint uniform_projection = glGetUniformLocation(programID, "projection");

	perspective(projection, M_PI/4.0, (float)width / (float)height, 1, 100.0f);
	lookAt(view, 0,0,20, 0,0,0, 0,1,0);

	//glUniform2f(uniform_WindowSize, width, height);
	glUniformMatrix4fv(uniform_model, 1, GL_FALSE, model);
	glUniformMatrix4fv(uniform_view, 1, GL_TRUE, view);
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

		lookAt(view, 20*cos(frame*M_PI/180.0),2,20*sin(frame*M_PI/180.0), 0,0,0, 0,1,0);
		glUniformMatrix4fv(uniform_view, 1, GL_TRUE, view);

		/*	Draw calls
		*/
		glDrawArrays(GL_TRIANGLES, 0, num_triangles[0]*3);


		/* per-loop stuff
		*/
		if(glGetError() != GL_NO_ERROR){
			fprintf(stderr, "Error in main loop\n");
			break;
		}

		// Control framerate
		frame++;
		/*if(current.tv_sec - previous_frame.tv_sec >= 1){
			fps = frame;
			frame = 0;
			previous_frame = current;
			printf("FPS: %u\n", fps);
		}*/

		glfwPollEvents();
		glfwSwapBuffers(window);
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window));

	// Cleanup
	free(triangles[0]);

	glDeleteBuffers(1, &color_buffer);
	glDeleteBuffers(1, &triangle_buffer);
	glfwDestroyWindow(window);
	glfwMakeContextCurrent(NULL);
	glfwTerminate();

	return 0;
}
