#include"plot.h"

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<GL/glut.h>

enum _plot_errors {NO_DATA = -1, NO_DIM = -2};

static uint32_t *HEIGHT;
static uint32_t *WIDTH;
static uint32_t *WINDOW_WIDTH;
static uint32_t *WINDOW_HEIGHT;

static struct _plot_pixel *data;

void setData(struct _plot_pixel *input){
	if(input == NULL){
		fprintf(stderr, "PLOT.setData: Input is null, not setting\n");
		return;
	}

	data = input;
//	printf("Data: %d\n", data[500].red);
}

static void resize(int new_width, int new_height){
	*WINDOW_WIDTH = new_width;
	*WINDOW_HEIGHT = new_height;
}

static void display(){// Main repainting/drawing function
	uint32_t W = *WIDTH;
	uint32_t H = *HEIGHT;
	float w_scale = (float)(*WINDOW_WIDTH)/W;
	float h_scale = (float)(*WINDOW_HEIGHT)/H;

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
//	printf("H: %d\tW: %d\tR = %f\tB = %f\n", H, W, w_scale, h_scale);
	gluOrtho2D(0.0 , W, H, 0.0);
	glViewport(0, 0, *WINDOW_WIDTH, * WINDOW_HEIGHT);

//	printf("Plotting..\n");
/*	if(W == *WINDOW_WIDTH && H == *WINDOW_HEIGHT){
		glBegin(GL_POINTS);
		for(uint32_t i = 0;i < W*H;i++){
//			printf("%d: (%hd,%hd,%hd) @ (%lu,%lu)\n", i, data[i].red, data[i].green, data[i].blue, data[i].x, data[i].y);
			glColor3f(data[i].red/255.0, data[i].green/255.0, data[i].blue/255.0);
			glVertex2i(data[i].x, H - data[i].y);
		}
		glEnd();
	}else{*/
		glLineWidth(1);
		for(uint32_t i = 0;i < W*H;i++){
			glBegin(GL_QUADS);
				glColor3f(data[i].red/255.0, data[i].green/255.0, data[i].blue/255.0);
				glVertex2f(data[i].x, H - data[i].y);
				glVertex2f(data[i].x+1, H - data[i].y);
				glVertex2f(data[i].x+1, H - data[i].y+1);
				glVertex2f(data[i].x, H - data[i].y+1);
			glEnd();
		}
//	}

	glFlush();// Draw screen
}

void plotInit(int width, int height, int x_offset, int y_offset){
//	glutInit(&argc, argv); // Required at beginning of main
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	data = NULL;

	glutInitWindowPosition(x_offset, y_offset);
	WIDTH = malloc(sizeof(*WIDTH));
	WINDOW_WIDTH = malloc(sizeof(*WINDOW_WIDTH));
	*WIDTH = width;
	*WINDOW_WIDTH = width;

	HEIGHT = malloc(sizeof(*HEIGHT));
	WINDOW_HEIGHT = malloc(sizeof(*WINDOW_HEIGHT));
	*HEIGHT = height;
	*WINDOW_HEIGHT = height;

	glutInitWindowSize(*WIDTH, *HEIGHT);

	glutCreateWindow("OpenGL Plot");

	glutDisplayFunc(display);
	glutReshapeFunc(resize);
//	glutMainLoop(); // Required at end of main
}

int plot(){ // Wrapper function for glutMainLoop()
	if(data == NULL){
		fprintf(stderr, "PLOT.plot: Data is null, cannot draw.\n");
		return NO_DATA;
	}else if(WIDTH == NULL || HEIGHT == NULL){
		fprintf(stderr, "PLOT.plot: Window not initialized\n");
		return NO_DIM;
	}

	glutMainLoop();
	return 0;
}
