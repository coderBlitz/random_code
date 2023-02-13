/* Conway's Game of Life
	Living cells:
		Lives if 2 or 3 neighbors
	Dead cells:
		Born if 3 neighbors

	Left-click and drag to move view
	Scroll in/out to zoom in or out (cannot scroll out more than original size, when it works)
	Right-click to reset view
	Press spacebar to pause/play
		When paused, press F to go to next generation (frame-by-frame)

	OpenGL notes:
	- Use single vertex buffer for squares
	- Repeat vertices for each square
	- Can maybe simplify vertex shader to not need to calc MVP (maybe)
	- figure out pan/zoom later
	TODO (option): May make instancing easier to draw every pixel, but color differently. Instead of offset array
		it would be a boolean array to color, and possibly a uniform value for the specific color. See how normal
		instancing performs (or not).
*/
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<string.h>
#include<unistd.h>
#include<math.h>
#include<time.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include"shader.c"

#ifdef GOL_USE_CL
#define CL_TARGET_OPENCL_VERSION 220
#define MAX_SOURCE_SIZE (0x10000)

#include<GL/glx.h> // For GLX calls for cl/gl mixing
#include<CL/cl.h>
#include<CL/cl_gl.h>
#endif

#define ACCESS2D(a,x,y) a[x*WIDTH + y]

// Side length of 2^15 should result in ~ 1GB of mem usage per grid
const uint32_t WIDTH = 1024; // Starting width and height
const uint32_t HEIGHT = 1024;
uint32_t WIN_WIDTH = WIDTH; // Current window width/height, changes with resize
uint32_t WIN_HEIGHT = HEIGHT;
const uint32_t LEVELS = 5; // Level(s) deal with zoom
int32_t LEVEL = LEVELS;
double XPOS = 0;
double XPOS_LAST; // Mouse positions
double YPOS = 0;
double YPOS_LAST;
char zoom_change = 0;
char game_pause = 1;
char frame_step = 0;


GLfloat square[] = {
	-1.0, -1.0,
	-1.0, 1.0,
	1.0, 1.0,
	1.0, 1.0,
	1.0, -1.0,
	-1.0, -1.0
};
size_t square_size = 12 * sizeof(*square);

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

struct generation{
	uint32_t population;
	char *grid;
};

void timespecsub(struct timespec *res, const struct timespec *a, const struct timespec *b){
	res->tv_sec = a->tv_sec - b->tv_sec;
	res->tv_nsec = a->tv_nsec - b->tv_nsec;
	if(res->tv_nsec < 0){
		res->tv_nsec += 1000000000;
		res->tv_sec -= 1;
	}
}

// Returns population. Assumes 'next' has been allocated in total
int nextGen(struct generation *cur, struct generation *next){
	if(cur == NULL || next == NULL){
		fprintf(stderr, "Please give pointers to nextGen.\n");
		return -1;
	}

	size_t population = 0;
	size_t count = 0; // Temporary count of living cells around given cell
	uint32_t prev_row = 0, next_row = 0;
	uint32_t prev_col = 0, next_col = 0;
	for(uint32_t row = 0;row < HEIGHT;row++){
		prev_row = (row - 1 + HEIGHT) % HEIGHT;
		next_row = (row + 1) % HEIGHT;

		for(uint32_t col = 0;col < WIDTH;col++){
			count = 0;
			prev_col = (col - 1 + WIDTH) % WIDTH;
			next_col = (col + 1) % WIDTH;

			count += cur->grid[prev_row * WIDTH + prev_col];
			count += cur->grid[prev_row * WIDTH + col];
			count += cur->grid[prev_row * WIDTH + next_col];

			count += cur->grid[row * WIDTH + prev_col];
			count += cur->grid[row * WIDTH + next_col];

			count += cur->grid[next_row * WIDTH + prev_col];
			count += cur->grid[next_row * WIDTH + col];
			count += cur->grid[next_row * WIDTH + next_col];

			//printf("row: %d\tcol: %d\tcount: %d\n", row, col, count);
			// If the cell is alive
			if(cur->grid[row * WIDTH + col]){
				if(count == 2 || count == 3){
					next->grid[row * WIDTH + col] = 1; // Stays alive
					population++;
				}else{
					next->grid[row * WIDTH + col] = 0; // Dies from over/underpopulation
				}
			}else{
				if(count == 3){
					next->grid[row * WIDTH + col] = 1; // Born
					population++;
				}else{
					next->grid[row * WIDTH + col] = 0; // Stays dead
				}
			}
		}
	}

	next->population = population;
	return population;
}

void printGrid(struct generation *grid){
	for(uint32_t row = 0;row < HEIGHT;row++){
		for(uint32_t col = 0;col < WIDTH;col++) printf("%c", (ACCESS2D(grid->grid, row, col))?'X':' ');
		printf("|\n");
	}
}

/** Scroll callback, changes zoom level.
**/
void zoom(GLFWwindow *window, double xoffset, double yoffset){
	double zoom_rate = 4.0 / 3.0;
	double zoom_rate_inv = 1.0 / zoom_rate;
	if((int)yoffset < 0){ // Zoom out (scale gets smaller)
		//LEVEL  = fmax(LEVEL - 1, LEVELS);
		//zoom_change = -1;
		projection[0] *= zoom_rate_inv;
		projection[5] *= zoom_rate_inv;

		//projection[12] *= zoom_rate_inv; // Should zoom on center of view
		//projection[13] *= zoom_rate_inv;
	}else{ // Zoom in (scale gets larger)
		//LEVEL += 1;
		//zoom_change = 1; // Toggle flag for display to center zoom, positive for in, negative for out
		projection[0] *= zoom_rate;
		projection[5] *= zoom_rate;

		//projection[12] *= zoom_rate; // Should zoom on center of view
		//projection[13] *= zoom_rate;
	}
	//printf("zoom: (%lf, %lf)\t%u/%u\n", xoffset, yoffset, LEVEL, LEVELS);
}

char left_down = 0;
void mousePos(GLFWwindow *window, double xpos, double ypos){
	XPOS_LAST = XPOS;
	YPOS_LAST = YPOS;
	XPOS = xpos;
	YPOS = ypos;

	double xdiff = XPOS - XPOS_LAST;
	double ydiff = YPOS - YPOS_LAST;
	float row_size = 1.0 / (HEIGHT); // Square height
	float col_size = 1.0 / (WIDTH); // Square width

	if(left_down){
		//projection[12] += xdiff * 2 * col_size; // 12 and 13 because of transpose for GL
		//projection[13] -= ydiff * 2 * row_size; // Shift only depends on original pixel size
		projection[3] += xdiff * 2.0 * col_size;
		projection[7] -= ydiff * 2.0 * row_size;
	}
}

void mouseButton(GLFWwindow *window, int button, int action, int mods){
	if(button == GLFW_MOUSE_BUTTON_LEFT){
		left_down = (action == GLFW_PRESS) ? 1 : 0;
		//printf("%s\n", (action == GLFW_PRESS)?"PRESS":"RELEASE");
	}else if(button == GLFW_MOUSE_BUTTON_RIGHT){
		LEVEL = LEVELS; // Reset zoom

		projection[0] = 1.0; // Reset zoom
		projection[5] = 1.0;

		projection[3] = 0.0; // Reset pan
		projection[7] = 0.0;
	}
}

void keyboard(GLFWwindow *window, int key, int scancode, int action, int mods){
	if(key == GLFW_KEY_SPACE && action == GLFW_PRESS) game_pause = !game_pause; // Stop at current generation
	if(key == GLFW_KEY_F && (action == GLFW_PRESS || action == GLFW_REPEAT)) frame_step = 1; // When paused, go to next generation
}

/**	Do all the openGL crap that I can split it
**/
GLFWwindow * initGL(void){
	if(!glfwInit()){
		fprintf(stderr, "GLFW init failed\n");
		return NULL;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // These cause basic drawing to not work, probably cause
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // the basics are deprecated or need shaders.
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	GLFWwindow *window;

	window = glfwCreateWindow(WIDTH, HEIGHT, "Conway's Game of Life!", NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Window failed!\n");
		glfwTerminate();
		return NULL;
	}
	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;

	if(glewInit() != GLEW_OK){
		fprintf(stderr, "Glew failed!\n");
		glfwTerminate();
		return NULL;
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSwapInterval(0);
	glfwSetScrollCallback(window, &zoom);
	//glfwSetCursorPosCallback(window, &mousePos); // Call back does not work for delta of mouse pos
	glfwSetMouseButtonCallback(window, &mouseButton);
	glfwSetKeyCallback(window, &keyboard);

	return window;
}


/***
		Main
***/
int main(int argc, char *argv[]){
	unsigned long target_fps = 120; // Computation rate
	long target_interval;
	// TODO: Add render framerate

	/* Parse cmd args
	*/
	long long_tmp;
	unsigned ulong_tmp;
	char *arg;
	for(long i = 1;i < argc;i++){
		if(argv[i][0] == '-'){
			switch(argv[i][1]){
				case 'l':
					arg = argv[++i];
					ulong_tmp = strtoul(argv[i], &arg, 10);
					if(arg == argv[i]){
						fprintf(stderr, "Unknown number: '%s'\n", argv[i]);
					}else{
						target_fps = ulong_tmp;
					}
					break;
				default:
					// Unknown flag
				break;
			}
		}
	}

	if(target_fps == 0){
		target_interval = 0; // No FPS limit
	}else{
		target_interval = 1000000000 / target_fps; // Target interval in nanoseconds
	}
	printf("Target interval: %lu\n", target_interval);

	/* GL initialization stuff
	*/
	GLFWwindow *window = initGL();
	if(window == NULL){
		fprintf(stderr, "Window failed!\n");
		glfwTerminate();
		return -1;
	}

	GLenum err;

	// Load program shaders
	GLuint programID = LoadShaders("vshader.glsl", "fshader.glsl");
	//printf("ProgramID: %u\n", programID);
	if(programID == 0){
		fprintf(stderr, "Program failed to load correctly.\n");
		glfwTerminate();
		return -3;
	}
	glUseProgram(programID);

	if((err = glGetError()) != GL_NO_ERROR){
		fprintf(stderr, "Error at prog: %X\n", err);
		return 1;
	}

#ifdef GOL_USE_CL
	/* Init openCL stuff
	*/
	printf("Init'ing openCL stuff..\n");
	// Get platform and device information
	const int max_platforms = 8;
	cl_platform_id platform_ids[max_platforms], active_platform = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(max_platforms, platform_ids, &ret_num_platforms);

	// Loop over platforms to grab first device that supports (if any)
	for(cl_uint id = 0;id < ret_num_platforms;id++){
		// Check for support
		char exts[1024];
		ret = clGetPlatformInfo(platform_ids[id], CL_PLATFORM_EXTENSIONS, sizeof(exts), exts, NULL);
		if(!strstr(exts, "cl_khr_gl_sharing")){
			printf("Platform %u (ID 0x%X) doesn't support CL GL sharing\n", id, platform_ids[id]);
		}else{
			active_platform = platform_ids[id];
			printf("Platform %u (ID 0x%X) supports CL GL sharing. Using..\n", id, platform_ids[id]);
			break;
		}
	}

	// Exit if no device supports
	if(active_platform == NULL){
		fprintf(stderr, "No platform supports CL-GL sharing. Exiting..\n");
		glfwTerminate();
		return -3;
	}

	// Fetch first device to use
	ret = clGetDeviceIDs(active_platform, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

	// Get properties, create context and queue
	cl_context_properties props[] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties) active_platform,
		CL_GLX_DISPLAY_KHR, (cl_context_properties) glXGetCurrentDisplay(),
		CL_GL_CONTEXT_KHR, (cl_context_properties) glXGetCurrentContext(),
		0
	};
	cl_context context = clCreateContext( props, 1, &device_id, NULL, NULL, &ret);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Context creation failed: %d\n", ret);
		glfwTerminate();
		return -2;
	}
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);

	// Load the opencl kernel source_str
	FILE *fp;
	char *source_str;
	size_t source_size;
	fp = fopen("evolve_kernel.cl", "r");
	if (fp == NULL) {
		fprintf(stderr, "Failed to load kernel.\n");
		glfwTerminate();
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Failed to build program\n");
		return -4;
	}
	free(source_str);

	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "nextGen", &ret);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Kernel creation failed\n");
		return -4;
	}
#endif

	// Only using 1 general set/type of data, so only one VAO
	GLuint VAO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	/*	Allocate everything for the boards
	*/
	struct generation a, b;
	struct generation *cur = &a, *next = &b;
	a.grid = calloc(WIDTH * HEIGHT, sizeof(*a.grid));
	if(a.grid == NULL){
		fprintf(stderr, "First grid failed to allocate\n");
		glfwTerminate();
		return -1;
	}
	b.grid = calloc(WIDTH * HEIGHT, sizeof(*b.grid));
	if(b.grid == NULL){
		fprintf(stderr, "Second grid failed to allocate\n");
		free(a.grid);
		glfwTerminate();
		return -1;
	}

	/***	Initialize the board with starting layout ***/
	/*/	Glider
	a.grid[0][2] = 1;
	a.grid[1][0] = 1;
	a.grid[1][2] = 1;
	a.grid[2][1] = 1;
	a.grid[2][2] = 1;
	*/
	/*/	Blinker
	a.grid[0][1] = 1;
	a.grid[1][1] = 1;
	a.grid[2][1] = 1;
	*/
	/*/ Beacon
	a.grid[4][1] = 1;
	a.grid[4][2] = 1;
	a.grid[4][3] = 1;
	a.grid[5][0] = 1;
	a.grid[5][1] = 1;
	a.grid[5][2] = 1;
	*/
	// Spaceship
	ACCESS2D(a.grid, 0, 0) = 1;
	ACCESS2D(a.grid, 0, 3) = 1;
	ACCESS2D(a.grid, 1, 4) = 1;
	ACCESS2D(a.grid, 2, 0) = 1;
	ACCESS2D(a.grid, 2, 4) = 1;
	ACCESS2D(a.grid, 3, 1) = 1;
	ACCESS2D(a.grid, 3, 2) = 1;
	ACCESS2D(a.grid, 3, 3) = 1;
	ACCESS2D(a.grid, 3, 4) = 1;
	/*a.grid[0][0] = 1;
	a.grid[0][3] = 1;
	a.grid[1][4] = 1;
	a.grid[2][0] = 1;
	a.grid[2][4] = 1;
	a.grid[3][1] = 1;
	a.grid[3][2] = 1;
	a.grid[3][3] = 1;
	a.grid[3][4] = 1;*/

	// Just a single row of cells
	for(int i = 0;i < WIDTH;i += 1) ACCESS2D(a.grid, 50, i) = 1;
	//printf("Starting grid:\n");
	//printGrid(&a);

	/*	Initialize offset data/matrices
	*/
	uint32_t total = HEIGHT * WIDTH;
	float row_size = 1.0 / (HEIGHT); // Square height
	float col_size = 1.0 / (WIDTH); // Square width

	printf("%d %f x %f squares\n", total , 2*col_size, 2*row_size);
	scale[0] = col_size;
	scale[5] = row_size;

	GLfloat *pixel_offsets;
	size_t pixel_offsets_size = 2 * total * sizeof(*pixel_offsets);
	pixel_offsets = malloc(pixel_offsets_size);
	if(pixel_offsets == NULL){
		fprintf(stderr, "Can't alloc offsets.\n");
		return -1;
	}

	// Calculate offsets
	for(int i = 0;i < HEIGHT;i += 1){
		for(int j = 0;j < WIDTH;j += 1){
			int idx = 2 * WIDTH * i + 2 * j;
			//pixel_offsets[idx] = 0;
			//pixel_offsets[idx+1] = 0;
			pixel_offsets[idx] = (((2.0 * j + 1.0) * col_size) - 1.0) / col_size; // Offset from left edge
			//pixel_offsets[idx+1] = (((2.0 * i + 1.0) * row_size) - 1.0) / row_size; // Offset from bottom edge
			pixel_offsets[idx+1] = (1.0 - ((2.0 * i + 1.0) * row_size)) / row_size; // Offset from top edge
		}
	}

	/*	Setup GL buffers
	*/
	GLuint square_vbo;
	glGenBuffers(1, &square_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, square_vbo);
	glBufferData(GL_ARRAY_BUFFER, square_size, square, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); // Drawing info
	glEnableVertexAttribArray(0);

	GLuint pixel_offsets_vbo;
	glGenBuffers(1, &pixel_offsets_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, pixel_offsets_vbo);
	glBufferData(GL_ARRAY_BUFFER, pixel_offsets_size, pixel_offsets, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0); // Drawing info
	glEnableVertexAttribArray(1);
	glVertexAttribDivisor(1, 1);

	GLuint cur_vbo;
	glGenBuffers(1, &cur_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, cur_vbo);
	glBufferData(GL_ARRAY_BUFFER, total, cur->grid, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(2, 1, GL_BYTE, GL_FALSE, 0, 0); // May need to re-call if flipping between two VBOs
	glEnableVertexAttribArray(2);
	glVertexAttribDivisor(2, 1);

	GLuint next_vbo;
	glGenBuffers(1, &next_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, next_vbo);
	glBufferData(GL_ARRAY_BUFFER, total, NULL, GL_DYNAMIC_DRAW); // Initialize empty buffer with same size

	if((err = glGetError()) != GL_NO_ERROR){
		fprintf(stderr, "Error in buffers: %X\n", err);
		return 1;
	}

	/*	More GL prep
	*/
	GLint uniform_WindowSize = glGetUniformLocation(programID, "WindowSize"); // Used to pass windows size
	GLuint uniform_model = glGetUniformLocation(programID, "model");
	GLuint uniform_view = glGetUniformLocation(programID, "view");
	GLuint uniform_projection = glGetUniformLocation(programID, "projection");

	glUniform2f(uniform_WindowSize, WIDTH, HEIGHT);
	glUniformMatrix4fv(uniform_model, 1, GL_TRUE, scale);
	glUniformMatrix4fv(uniform_view, 1, GL_TRUE, rotation);
	glUniformMatrix4fv(uniform_projection, 1, GL_TRUE, projection);

	if((err = glGetError()) != GL_NO_ERROR){
		fprintf(stderr, "Error in main: %X\n", err);
		return 1;
	}

	/* Stuff before loop
	*/
#ifdef GOL_USE_CL
	// Create memory buffers on the device for each VBO
	cl_mem cur_vbo_cl = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, cur_vbo, &ret);
	cl_mem next_vbo_cl = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, next_vbo, &ret);
	cl_mem width_cl = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(WIDTH), NULL, &ret);

	// Flipping pointers (literally)
	cl_mem *cur_cl = &cur_vbo_cl, *next_cl = &next_vbo_cl, *tmp_cl;

	// Write width value
	ret = clEnqueueWriteBuffer(command_queue, width_cl, CL_TRUE, 0, sizeof(WIDTH), (const void *)&WIDTH, 0, NULL, NULL);

	// Set unchanged kernel arg for width value
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&width_cl);

	// Kernel size parameters
	size_t global_item_size = WIDTH*HEIGHT;
	size_t global_work_size = 1024; // 64 max for weak GPU, 1024 for good
#endif

	//glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_LESS);
	glBindBuffer(GL_ARRAY_BUFFER, cur_vbo); // To make CPU only work

	/*** Start the evolution and display the window
	***/
	struct timespec current, previous;
	struct timespec frame_ts, diff;
	struct timespec draw_beg, draw_end, up_beg, up_end;
	uint16_t fps = 0;
	register uint64_t frame = 0;
	clock_gettime(CLOCK_REALTIME, &current);
	frame_ts = current;
	previous = current;

	double xpos, ypos;
	struct generation *gen_tmp;
	GLuint *vbo_cur = &cur_vbo, *vbo_next = &next_vbo, *vbo_tmp;
	do{
		clock_gettime(CLOCK_REALTIME, &current); // Start frame time

		/*	Draw
		*/
		glClearColor(0.0f, 0.2f, 0.0f, 0.0f); // Set background color
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glDrawArraysInstanced(GL_TRIANGLES, 0, 2*3, total); // Draw 'total' squares, with 6 vertices each, starting index 0

		clock_gettime(CLOCK_REALTIME, &draw_end); // Time profiling
		draw_beg = current;

		if((err = glGetError()) != GL_NO_ERROR){
			fprintf(stderr, "Error after drawing: %X\n", err);
			break;
		}

		/* Compute and update the next generation
		*/
		// Compute next generation, if unpaused, or if frame stepping
		clock_gettime(CLOCK_REALTIME, &up_beg);
		if(!game_pause){
			// Check if time to last frame has hit the interval, otherwise don't compute next
			timespecsub(&diff, &current, &frame_ts);
			long_tmp = target_interval - (diff.tv_sec * 1000000000 + diff.tv_nsec);

			// If passed target interval, calculate next frame.
			if(long_tmp <= 0){
				goto frame_next;
			}
		}else if(frame_step){ // Step frame by one
			frame_step = 0;

		frame_next:
#ifdef GOL_USE_CL
			// Set args
			ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)cur_cl);
			if(ret != CL_SUCCESS){
				fprintf(stderr, "Kernel arg failed: %d\n", ret);
				break;
			}
			clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)next_cl);

			// Acquire GL buffers
			ret = clEnqueueAcquireGLObjects(command_queue, 1, &cur_vbo_cl, 0, NULL, NULL);
			if(ret){
				printf("Could not acquire GL cur object: %d\n", ret);
				if(ret == CL_INVALID_CONTEXT) printf("Context\n");
				break;
			}
			ret = clEnqueueAcquireGLObjects(command_queue, 1, &next_vbo_cl, 0, NULL, NULL);
			if(ret){
				printf("Could not acquire GL next object: %d\n", ret);
				if(ret == CL_INVALID_CONTEXT) printf("Context\n");
				break;
			}

			// Compute next frame
			ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &global_work_size, 0, NULL, NULL);
			if(ret != CL_SUCCESS){
				fprintf(stderr, "Kernel failed: %d\n", ret);
				break;
			}

			// Release GL objects
			ret = clEnqueueReleaseGLObjects(command_queue, 1, &cur_vbo_cl, 0, NULL, NULL);
			ret = clEnqueueReleaseGLObjects(command_queue, 1, &next_vbo_cl, 0, NULL, NULL);

			// Swap CL stuff
			tmp_cl = cur_cl;
			cur_cl = next_cl;
			next_cl = tmp_cl;

			// Swap VBOs
			vbo_tmp = vbo_cur;
			vbo_cur = vbo_next;
			vbo_next = vbo_tmp;

			// Update the GL data buffer pointer
			glBindBuffer(GL_ARRAY_BUFFER, *vbo_cur);
			glVertexAttribPointer(2, 1, GL_BYTE, GL_FALSE, 0, 0);
#else
			// Host (CPU) evolve
			nextGen(cur, next);

			// Swap current and next (or previous and current)
			gen_tmp = cur;
			cur = next;
			next = gen_tmp;

			// Update buffer data
			glBufferData(GL_ARRAY_BUFFER, total, NULL, GL_DYNAMIC_DRAW);
			glBufferSubData(GL_ARRAY_BUFFER, 0, total, cur->grid);
#endif

			// Update last frame time
			frame++;
			clock_gettime(CLOCK_REALTIME, &frame_ts); // End frame time
		}
		clock_gettime(CLOCK_REALTIME, &up_end);

		/* per-loop stuff
		*/
		glfwPollEvents();
		glfwSwapBuffers(window);
		//glfwGetWindowSize(window, &WIN_WIDTH, &WIN_HEIGHT);

		glfwGetCursorPos(window, &xpos, &ypos);
		mousePos(window, xpos, ypos);
		//glUniformMatrix4fv(uniform_model, 1, GL_FALSE, scale); // For updated scale
		glUniformMatrix4fv(uniform_projection, 1, GL_TRUE, projection);
		//printf("(%lf, %lf)\n", xpos, ypos);

		// Track framerate
		if(current.tv_sec - previous.tv_sec >= 1){
			fps = frame;
			frame = 0;
			previous = current;
			printf("FPS: %3u\tDraw: %8dns\tNext gen: %10dns\n", fps, draw_end.tv_nsec - draw_beg.tv_nsec, up_end.tv_nsec - up_beg.tv_nsec);
		}

		/* Control render framerate
		*/
		long_tmp = target_interval - (frame_ts.tv_sec * 1000000000 + frame_ts.tv_nsec); // Target time subtract frame time

		// Sleep for remaining difference
		/*if(long_tmp > 0){
			frame_ts.tv_sec = 0;
			frame_ts.tv_nsec = long_tmp;
			nanosleep(&frame_ts, NULL);
		}*/
		//usleep(250000);
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);
	printf("\n");

	/***	Clean up ***/
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &next_vbo);
	glDeleteBuffers(1, &cur_vbo);
	glDeleteBuffers(1, &pixel_offsets_vbo);
	glDeleteBuffers(1, &square_vbo);
	free(pixel_offsets);

#ifdef GOL_USE_CL
	clReleaseMemObject(width_cl);
	clReleaseMemObject(next_vbo_cl);
	clReleaseMemObject(cur_vbo_cl);
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
#endif

	glfwDestroyWindow(window);
	glfwTerminate();

	free(a.grid);
	free(b.grid);

	return 0;
}
