/* Conway's Game of Life
	Living cells:
		Lives if 2 or 3 neighbors
	Dead cells:
		Born if 3 neighbors

	Left-click and drag to move view
	Scroll in/out to zoom in or out (cannot scroll out more than original size)
	Right-click to reset view
	Press spacebar to pause/play
		When paused, press F to go to next generation (frame-by-frame)

	OpenGL notes:
	- Use single vertex buffer for squares
	- Repeat vertices for each square
	- Can maybe simplify vertex shader to not need to calc MVP (maybe)
	- figure out pan/zoom later
	TODO: Add GL buffers
	TODO: Add transormation matrices
	TODO: Use GL instance rendering for repeated vertices/textures
*/
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<unistd.h>
#include<math.h>
#include<time.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include"shader.c"

// Side length of 2^15 should result in ~ 1GB of mem usage per grid
const uint32_t WIDTH = 500; // Starting width and height
const uint32_t HEIGHT = 500;
uint32_t WIN_WIDTH = WIDTH; // Current window width/height, changes with resize
uint32_t WIN_HEIGHT = HEIGHT;
const uint32_t LEVELS = 5; // Level(s) deal with zoom
int32_t LEVEL = LEVELS;
double XPOS = 0;
double XPOS_LAST; // Mouse positions
double YPOS = 0;
double YPOS_LAST;
double cornerx = 0; // Lower left corner of view box
double cornery = 0;
double offsetx = 0; // Offset determined by mouse movement
double offsety = 0;
double screen_centerx = 0; // Center of viewing window in absolute coordinates
double screen_centery = 0;
char zoom_change = 0;
char game_pause = 0;
char frame_step = 0;


int8_t square[] = {
	-1.0f, -1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	1.0f, 1.0f, 0.0f,
	-1.0f, 1.0f, 0.0f
};

struct generation{
	uint32_t population;
	char **grid;
};

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
				if(count == 2 || count == 3){
					next->grid[row][col] = 1; // Stays alive
					population++;
				} else next->grid[row][col] = 0; // Dies from over/underpopulation
			}else{
				if(count == 3){
					next->grid[row][col] = 1; // Born
					population++;
				}else next->grid[row][col] = 0; // Stays dead
			}
		}
	}

	next->population = population;
	return population;
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

void keyboard(GLFWwindow *window, int key, int scancode, int action, int mods){
	if(key == GLFW_KEY_SPACE && action == GLFW_PRESS) game_pause = !game_pause; // Stop at current generation
	if(key == GLFW_KEY_F && action == GLFW_PRESS) frame_step = 1; // When paused, go to next generation
}

struct generation a, b;
double zoom_factor;
double VIEW_WIDTH;
double VIEW_HEIGHT;
double mouse_relx;
double mouse_rely;
double dx;
double dy;
void display(struct generation *cur, struct generation *next){
	glClearColor(0.0f, 0.2f, 0.0f, 0.0f); // Set background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLineWidth(1.0);
	// Draw all living cells
	for(uint32_t row = 0;row < HEIGHT;row++){
		for(uint32_t col = 0;col < WIDTH;col++){
			if(cur->grid[row][col]){
			}
		}
	}
}

GLFWwindow * initGL(){
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
	glfwSwapInterval(1);
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
	GLFWwindow *window = initGL();
	if(window == NULL){
		fprintf(stderr, "Window failed!\n");
		glfwTerminate();
		return -1;
	}

	// Prepare for loop
	GLuint programID = LoadShaders("vshader.glsl", "fshader.glsl");
	//printf("ProgramID: %u\n", programID);
	if(programID == 0){
		fprintf(stderr, "Program failed to load correctly.\n");
		glfwTerminate();
		return -3;
	}

	struct timespec current, previous;

	/***	Allocate everything for the boards ***/
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

	// Allocate each row
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
	a.grid[0][0] = 1;
	a.grid[0][3] = 1;
	a.grid[1][4] = 1;
	a.grid[2][0] = 1;
	a.grid[2][4] = 1;
	a.grid[3][1] = 1;
	a.grid[3][2] = 1;
	a.grid[3][3] = 1;
	a.grid[3][4] = 1;

	// Just a single row of cells
	for(int i = 0;i < WIDTH;i += 1) a.grid[50][i] = 1;
	//printf("Starting grid:\n");
	//printGrid(&a);

	/***	Start the evolution and display the window ***/
	uint16_t fps = 0;
	register uint64_t frame = 0;
	clock_gettime(CLOCK_REALTIME, &current);
	previous = current;
	double xpos, ypos;
	do{
		clock_gettime(CLOCK_REALTIME, &current);
		display(cur, next);

		if(!game_pause){
			nextGen(cur, next);

			//printf("Next gen:\n");
			//printGrid(next);

			// Swap current and next (or previous and current)
			cur = (void *)((uintptr_t) cur ^ (uintptr_t) next);
			next = (void *)((uintptr_t) next ^ (uintptr_t) cur);
			cur = (void *)((uintptr_t) next ^ (uintptr_t) cur);	
		}else if(frame_step){ // Step frame by one
			nextGen(cur, next);

			cur = (void *)((uintptr_t) cur ^ (uintptr_t) next);
			next = (void *)((uintptr_t) next ^ (uintptr_t) cur);
			cur = (void *)((uintptr_t) next ^ (uintptr_t) cur);

			frame_step = 0;
		}

		// Do graphics repeat stuff
		glfwSwapBuffers(window);
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
			printf("\rFPS: %u", fps);
			fflush(stdout);
		}
		//printf("\rTime: %lus %luns", current.tv_sec - previous.tv_sec, current.tv_nsec);
		usleep(25000);
	}while(glfwWindowShouldClose(window) == 0);
	printf("\n");

	/***	Clean up ***/
	glfwDestroyWindow(window);
	glfwTerminate();

	for(uint32_t i = 0;i < HEIGHT;i++){
		free(a.grid[i]);
		free(b.grid[i]);
	}
	free(a.grid);
	free(b.grid);
}
