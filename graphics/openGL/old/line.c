#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<GL/glut.h>

static int X=1;

void display(){// Main repainting/drawing function
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glLineWidth(2.5);
	glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_LINES);
	glVertex2f(0.5, 0.5);
	glVertex2f(0.25, 0.25);
	glEnd();

	glLineWidth(2.5);
	glColor3f(0.0, 0.0, 1.0);
	glBegin(GL_LINES);
	glVertex2f(0.5, 0.25);
	glVertex2f(0.25, 0.5);
	glEnd();

	glFlush();// Draw screen
}

void timer_event(int te){
	X++;

	if(X == 50){}
	else{
		glutPostRedisplay();
		glutTimerFunc(50,timer_event,1);// Re-calls itself until X == 50
	}
}

int main(int argc, char **argv){
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_SINGLE|GLUT_RGBA);
	glutInitWindowPosition(90,0);
	glutInitWindowSize(500,500);
	glutCreateWindow("LINES EVERYWHERE!!!");

	glutDisplayFunc(display);

	printf("Wait for it...\n");
//	glutTimerFunc(50,timer_event,1);// Calls function in 50ms

	glutMainLoop();
}
