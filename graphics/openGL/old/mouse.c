#include<stdio.h>
#include<stdlib.h>
#include<GL/glut.h>

#define WIDTH 600
#define HEIGHT 500

static int mouse_x;
static int mouse_y;
static char draw;

struct _xy_coord {
	uint32_t x;
	uint32_t y;
};

static struct _xy_coord *points; // Store all drawn points
static uint32_t count;


void finale(){
	free(points);
}

void display(){
	glClearColor(0.0, 0.0, 0.0, 0.0); // Clear then set background color (RGB(A?))
	glClear(GL_COLOR_BUFFER_BIT);

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

	glFlush();
}

#define LEFT_CLICK 			0
#define MIDDLE_CLICK		1
#define RIGHT_CLICK		2
#define SCROLL_UP			3
#define SCROLL_DOWN		4
#define PRESSED				0
#define RELEASED				1

void clearPoints(int butt, int state, int x, int y){
//	printf("Button: %d\tState: %d\n", butt, state);

	if(butt == RIGHT_CLICK && state == PRESSED){
		free(points);
		points = NULL; // Kind of important
		count = 0; // Mostly important
		draw = 0;
	}else if(butt == LEFT_CLICK && state == PRESSED){
//		printf("Draw: %d\n", draw);
		draw = 1;
	}else if(state == PRESSED){
		printf("Useless button\n");
		draw = 0;
	}

	glutPostRedisplay();
}

void updatePoints(int x, int y){
//	printf("X: %d\tY: %d\n", x, y);
	mouse_x = x;
	mouse_y = y;

	if(draw){
		count += 1;
		if(points == NULL){
			points = malloc(sizeof(*points));
			points[0].x = x;
			points[0].y = y;
		}else{
			points = realloc(points, count * sizeof(*points));
			points[count-1].x = x;
			points[count-1].y = y;
		}
	}

	glutPostRedisplay();
}

int main(int argc, char *argv[]){
	atexit(finale);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

	mouse_x = -1;
	mouse_y = -1;
	count = 0;
	draw = 0;
	points = NULL;

	glutInitWindowPosition(80,80);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Mouse test");

	glutDisplayFunc(display);
	glutMotionFunc(updatePoints);
	glutMouseFunc(clearPoints);

	glutMainLoop();
	return 0;
}
