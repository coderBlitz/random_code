#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>

static int mouse_x;
static int mouse_y;
static char draw;

struct _xy_coord {
	uint32_t x;
	uint32_t y;
};
static struct _xy_coord *points; // Store all drawn points
static uint32_t count;

void display(){
	glMatrixMode(GL_PROJECTION); // Working with projection matrix
	glLoadIdentity(); // Make current matrix the identity matrix
	gluOrtho2D(0.0, WIDTH, HEIGHT, 0.0); // left, right, bottom, top clipping planes

	if(count > 0){
		glBegin(GL_POINTS);
		for(uint32_t i =0;i < count;i++){
			glColor3f(1.0, 1.0, 1.0);
			glVertex2i(points[i].x, points[i].y);
		}
		glEnd();
	}
}

void updatePoints(){
	
}

int main(int argc, char *argv[]){
	if(!glfwInit()){
		fprintf(stderr, "GLFW init failed\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window;

	window = glfwCreateWindow(1024, 768, "Title Bar name", NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Window creation failed.\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glewExperimental = 1;

	if(glewInit() != GLEW_OK){
		fprintf(stderr, "GLEW init failed.\n");
		glfwTerminate();
		return -1;
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	do{
		glClear( GL_COLOR_BUFFER_BIT );

		updatePoints();
		display();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}while(glfwWindowShouldClose(window) == 0);

	glfwTerminate();

	free(points);
	return 0;
}
