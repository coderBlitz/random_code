/* Conway's Game of Life
	Living cells:
		Lives if 2 or 3 neighbors
	Dead cells:
		Born if 3 neighbors
*/
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<unistd.h>
#include<math.h>
#include<time.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include<CL/cl.h>

#define MAX_SOURCE_SIZE (0x1000)

// Side length of 2^15 should result in ~ 1GB of mem usage per grid
const uint32_t WIDTH = 1024;
const uint32_t HEIGHT = 1024;
uint32_t WIN_WIDTH = WIDTH;
uint32_t WIN_HEIGHT = HEIGHT;
const uint32_t LEVELS = 5; // Level(s) deal with zoom
int32_t LEVEL = LEVELS;
double XPOS = 0;
double XPOS_LAST; // Mouse positions
double YPOS = 0;
double YPOS_LAST;
double cornerx = 0; // Lower left corner of view box
double cornery = 0;
double offsetx = 0;
double offsety = 0;
double screen_centerx = 0;
double screen_centery = 0;
char zoom_change = 0;

struct generation{
	uint32_t population;
	char **grid;
};
struct generation a, b;

// Returns population. Assumes 'next' has been allocated in total
int nextGen(struct generation *cur, struct generation *next){
	if(cur == NULL || next == NULL){
		fprintf(stderr, "Please give pointers to nextGen.\n");
		return -1;
	}

	uint32_t population = 0;
	uint32_t count = 0; // Temporary count of living cells around given cell
	uint16_t prev_row = 0, next_row = 0;
	uint16_t prev_col = 0, next_col = 0;
	for(uint32_t row = 0;row < HEIGHT;row++){
		prev_row = (row - 1 + HEIGHT) % HEIGHT;
		next_row = (row + 1) % HEIGHT;
		for(uint32_t col = 0;col < WIDTH;col++){
			count = 0;
			prev_col = (col - 1 + WIDTH) % WIDTH;
			next_col = (col + 1) % WIDTH;

			count += cur->grid[prev_row][prev_col];
			count += cur->grid[prev_row][col];
			count += cur->grid[prev_row][next_col];

			count += cur->grid[row][prev_col];
			count += cur->grid[row][next_col];

			count += cur->grid[next_row][prev_col];
			count += cur->grid[next_row][col];
			count += cur->grid[next_row][next_col];

			//printf("row: %d\tcol: %d\tcount: %d\n", row, col, count);
			// If the cell is alive
			if(cur->grid[row][col]){
				if(count == 2 || count == 3) next->grid[row][col] = 1;
				else next->grid[row][col] = 0; // Dies from over/underpopulation
			}else{
				if(count == 3){
					next->grid[row][col] = 1;
					//printf("(%d,%d) is born!\n", row, col);
				}else next->grid[row][col] = 0;
			}
		}
	}	glEnd();

}

void printGrid(struct generation *grid){
	for(uint32_t row = 0;row < HEIGHT;row++){
		for(uint32_t col = 0;col < WIDTH;col++) printf("%c", (grid->grid[row][col])?'X':' ');
		printf("|\n");
	}
}

void zoom(GLFWwindow *window, double xoffset, double yoffset){
	if((int)yoffset < 0){
		LEVEL  = fmax(LEVEL - 1, LEVELS);
		zoom_change = -1;
	}else{
		LEVEL += 1;
		zoom_change = 1; // Toggle flag for display to center zoom, positive for in, negative for out
	}
	//printf("zoom: (%lf, %lf)\t%u/%u\n", xoffset, yoffset, LEVEL, LEVELS);
}

void mousePos(GLFWwindow *window, double xpos, double ypos){
	XPOS_LAST = XPOS;
	YPOS_LAST = YPOS;
	XPOS = xpos;
	YPOS = ypos;
}

char left_down = 0;
void mouseButton(GLFWwindow *window, int button, int action, int mods){
	if(button == GLFW_MOUSE_BUTTON_LEFT){
		left_down = (action == GLFW_PRESS)?1:0;
		//printf("%s\n", (action == GLFW_PRESS)?"PRESS":"RELEASE");
	}else if(button == GLFW_MOUSE_BUTTON_RIGHT){
		LEVEL = LEVELS; // Reset zoom
		cornerx = 0;
		cornery = 0;
		offsetx = 0;
		offsety = 0;
	}
}

double zoom_factor;
double VIEW_WIDTH;
double VIEW_HEIGHT;
double mouse_relx;
double mouse_rely;
double dx;
double dy;
void display(struct generation *cur, struct generation *next){
	glClearColor(0.0f, 0.2f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity(); // Make current matrix the identity matrix
	glMatrixMode(GL_PROJECTION); // Working with projection matrix

	zoom_factor = (float)LEVEL / (float)LEVELS;
	VIEW_WIDTH = WIN_WIDTH*zoom_factor;
	VIEW_HEIGHT = WIN_HEIGHT*zoom_factor;
	mouse_relx = cornerx + XPOS/zoom_factor;
	mouse_rely = cornery + YPOS/zoom_factor;
	screen_centerx = WIN_WIDTH/2 - offsetx/zoom_factor;
	screen_centery = WIN_HEIGHT/2 + offsety/zoom_factor;

	if(left_down){
		// Do screen movement stuff
		dx = XPOS - XPOS_LAST;
		dy = YPOS - YPOS_LAST;

		offsetx += dx;
		offsety -= dy;
	}

	if(zoom_change){
		//printf("X: %lf\tY: %lf\n", screen_centerx, screen_centery);
		cornerx = -(VIEW_WIDTH - WIN_WIDTH)/2.0;
		cornery = -(VIEW_HEIGHT - WIN_HEIGHT)/2.0;

		offsetx += zoom_change * offsetx / (double)(zoom_factor * LEVELS);
		offsety += zoom_change * offsety / (double)(zoom_factor * LEVELS);
		zoom_change = 0;
	}
	/*printf("(%.3lf, %.3lf)+(%.3lf, %.3lf)>(%.3lf, %.3lf)\t[%.3lf, %.3lf]\n"
		, cornerx
		, cornery
		, offsetx
		, offsety
		, screen_centerx
		, screen_centery
		, VIEW_HEIGHT
		, VIEW_WIDTH);
	*/

	glViewport(cornerx + offsetx, cornery + offsety, VIEW_WIDTH, VIEW_HEIGHT);
	//glViewport(screen_centerx - WIN_WIDTH/2.0, screen_centery - WIN_HEIGHT/2.0, VIEW_WIDTH, VIEW_HEIGHT);
	glOrtho(0.0, WIDTH, HEIGHT,  0.0, 100, -100); // left, right, bottom, top clipping planes

	glLineWidth(1.0);
	glPointSize(1);

	for(uint32_t row = 0;row < HEIGHT;row++){
	glBegin(GL_QUADS);
		for(uint32_t col = 0;col < WIDTH;col++){
			if(cur->grid[row][col]){
				glColor3f(1.0, 1.0, 1.0);
				glVertex2i(col, row);
				glVertex2i(col+1, row);
				glVertex2i(col+1, row+1);
				glVertex2i(col, row+1);
			}
		}
		glEnd();
	}

	/*
	glBegin(GL_QUADS);
	glColor3f(0.0, 1.0, 1.0);
	glVertex2i(- offsetx/zoom_factor + WIN_WIDTH/2, offsety/zoom_factor + WIN_HEIGHT/2);
	glVertex2i(- offsetx/zoom_factor + WIN_WIDTH/2+1, offsety/zoom_factor + WIN_HEIGHT/2);
	glVertex2i( - offsetx/zoom_factor + WIN_WIDTH/2+1, offsety/zoom_factor + WIN_HEIGHT/2+1);
	glVertex2i( - offsetx/zoom_factor + WIN_WIDTH/2, offsety/zoom_factor + WIN_HEIGHT/2+1);
	glEnd();
	*/
	glPointSize(2);
	glBegin(GL_POINTS);
	glColor3f(1.0, 1.0, 0.0);
	glVertex2f(screen_centerx, screen_centery);
	glEnd();

	glFlush();
}

GLFWwindow * initGL(){
	if(!glfwInit()){
		fprintf(stderr, "GLFW init failed\n");
		return NULL;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	//glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // These cause basic drawing to not work, probably cause
	//glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // the basics are deprecated or need shaders.
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
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

	return window;
}

int main(int argc, char *argv[]){
	FILE *fp = fopen("evolve_kernel.cl", "r");
	if(fp == NULL){
		fprintf(stderr, "evolve_kernel.cl not found, cannot continue\n");
		return -3;
	}
	char *source = malloc(MAX_SOURCE_SIZE * sizeof(*source));
	size_t source_size = 0;
	source_size = fread(source, sizeof(*source), MAX_SOURCE_SIZE, fp);
	fclose(fp);

	GLFWwindow *window = initGL();
	if(window == NULL){
		fprintf(stderr, "Window failed!\n");
		glfwTerminate();
		return -1;
	}

	// Initialize OpenCL stuff
	cl_platform_id plat;
	cl_device_id dev;
	cl_int ret;
	cl_uint num_devs;
	cl_uint num_plats;
	size_t global_item_size = WIDTH*HEIGHT;
	size_t global_work_size = 64;

	ret = clGetPlatformIDs(1, &plat, &num_plats);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Platform failed\n");
		return -7;
	}
	ret = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &num_devs);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Device failed\n");
		return -7;
	}

	cl_context context = clCreateContext(NULL, 1, &dev, NULL, NULL, &ret);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Thing failed: %d\n", ret);
		return -6;
	}
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, dev, NULL, &ret);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Thing failed: %d\n", ret);
		return -6;
	}

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, &source_size, &ret);
	ret = clBuildProgram(program, 1, &dev, NULL, NULL, NULL);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Failed to build program\n");
		return -4;
	}
	cl_kernel kernel = clCreateKernel(program, "nextGen", &ret);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Kenerl creation failed\n");
		return -4;
	}

	cl_mem cur_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, global_item_size * sizeof(char), NULL, &ret);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Failed to allocate device new buffer\n");

		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		glfwDestroyWindow(window);
		glfwTerminate();
		return -2;
	}
	cl_mem next_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, global_item_size * sizeof(**b.grid), NULL, &ret);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Failed to allocate device next buffer\n");

		clReleaseMemObject(cur_mem);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		glfwDestroyWindow(window);
		glfwTerminate();
		return -2;
	}
	cl_mem width_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(WIDTH), NULL, &ret);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Failed to allocate device width buffer\n");

		clReleaseMemObject(next_mem);
		clReleaseMemObject(cur_mem);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		glfwDestroyWindow(window);
		glfwTerminate();
		return -2;
	}
	ret = clEnqueueWriteBuffer(queue, width_mem, CL_TRUE, 0, sizeof(WIDTH), (const void *)&WIDTH, 0, NULL, NULL);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Failed to copy width\n");
		return -5;
	}
	// End OpenCL stuff

	struct timespec current, previous;

	// Allocate everything for the board
	struct generation *cur = &a, *next = &b;
	a.grid = calloc(HEIGHT, sizeof(*a.grid));
	if(a.grid == NULL){
		fprintf(stderr, "First grid failed to allocate\n");
		glfwTerminate();
		return -1;
	}
	b.grid = calloc(HEIGHT, sizeof(*b.grid));
	if(b.grid == NULL){
		fprintf(stderr, "Second grid failed to allocate\n");
		free(a.grid);
		glfwTerminate();
		return -1;
	}

	for(uint32_t i = 0;i < HEIGHT;i++){
		a.grid[i] = calloc(WIDTH, sizeof(**a.grid));
		if(a.grid[i] == NULL){
			fprintf(stderr, "(%d, *) failed to allocate\n", i);
			for(uint32_t j = i-1; i >= 0;i--){
				free(a.grid[j]);
				free(b.grid[j]);
			}
			free(a.grid);
			free(b.grid);
			glfwTerminate();
			return -1;
		}

		b.grid[i] = calloc(WIDTH, sizeof(**b.grid));
		if(b.grid[i] == NULL){
			fprintf(stderr, "(%d, *) failed to allocate\n", i);
			free(a.grid[i]);
			for(uint32_t j = i-1; i >= 0;i--){
				free(a.grid[j]);
				free(b.grid[j]);
			}
			free(a.grid);
			free(b.grid);
			glfwTerminate();
			return -1;
		}
	}

	// Initialize the board with starting layout
	/*/ Glider
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
	a.grid[0][0] = 1;
	a.grid[0][3] = 1;
	a.grid[1][4] = 1;
	a.grid[2][0] = 1;
	a.grid[2][4] = 1;
	a.grid[3][1] = 1;
	a.grid[3][2] = 1;
	a.grid[3][3] = 1;
	a.grid[3][4] = 1;

	for(int i = 0;i < WIDTH;i += 1) a.grid[50][i] = 1;
	printf("Starting grid:\n");
	//printGrid(&a);

	// Copy grids to CL buffers
	printf("Copying grid to device.. ");
	for(int i = 0;i < HEIGHT;++i){
		ret = clEnqueueWriteBuffer(queue, cur_mem, CL_FALSE, i*WIDTH, WIDTH, a.grid[i], 0, NULL, NULL);
		if(ret != CL_SUCCESS){
			fprintf(stderr, "Failed to copy row %d to device\n", i);
		}
	}
	clFinish(queue);
	printf("Done\n");

	// Start the evolution and display the window
	uint16_t fps = 0;
	register uint64_t frame = 0;
	clock_gettime(CLOCK_REALTIME, &current);
	previous = current;
	double xpos, ypos;
	cl_mem temp;
	struct generation *gen_tmp;
	do{
		clock_gettime(CLOCK_REALTIME, &current);
		display(cur, next);
		//nextGen(cur, next);

		// Calculate multiple per render
		for(int i = 0;i < 1;i++){
			ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cur_mem);
			if(ret != CL_SUCCESS){
				fprintf(stderr, "Kernel arg failed: %d\n", ret);
				break;
			}
			clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&next_mem);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&width_mem);
			ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size, &global_work_size, 0, NULL, NULL);
			if(ret != CL_SUCCESS){
				fprintf(stderr, "Kernel failed: %d\n", ret);
				break;
			}
			for(int i = 0;i < HEIGHT;++i){
				ret = clEnqueueReadBuffer(queue, next_mem, CL_FALSE, i*WIDTH, WIDTH, next->grid[i], 0, NULL, NULL);
				if(ret != CL_SUCCESS){
					fprintf(stderr, "Failed to copy row %d to device\n", i);
				}
			}

			// Swap current and next (or previous and current)
			/*cur = (void *)((uintptr_t) cur ^ (uintptr_t) next);
			next = (void *)((uintptr_t) next ^ (uintptr_t) cur);
			cur = (void *)((uintptr_t) next ^ (uintptr_t) cur);*/
			gen_tmp = cur;
			cur = next;
			next = gen_tmp;

			temp = cur_mem;
			cur_mem = next_mem;
			next_mem = temp;
		}

		//printf("Next gen:\n");
		//printGrid(next);
		//printf("\n");

		// Do graphics repeat stuff
		glfwSwapBuffers(window);
		glfwSwapInterval(0);
		glfwPollEvents();
		glfwGetWindowSize(window, &WIN_WIDTH, &WIN_HEIGHT);

		glfwGetCursorPos(window, &xpos, &ypos);
		mousePos(window, xpos, ypos);
		//printf("(%lf, %lf)\n", xpos, ypos);

		// Control framerate
		frame++;
		if(current.tv_sec - previous.tv_sec >= 1){
			fps = frame;
			frame = 0;
			previous = current;
			//printf("\rFPS: %u", fps);
			//fflush(stdout);
			printf("FPS: %u\n", fps);
		}
		//printf("\rTime: %lus %luns", current.tv_sec - previous.tv_sec, current.tv_nsec);
		//usleep(12500);
		clFinish(queue);
	}while(glfwWindowShouldClose(window) == 0);
	main_loop_end:
	printf("\n");

	// Clean up
	glfwDestroyWindow(window);
	glfwTerminate();

	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(cur_mem);
	clReleaseMemObject(next_mem);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	for(uint32_t i = 0;i < HEIGHT;i++){
		free(a.grid[i]);
		free(b.grid[i]);
	}
	free(a.grid);
	free(b.grid);
	free(source);
}
