#include<stdio.h>
#include<math.h>
#include<GL/glut.h>

static int degree=0;

void display(){
	glClearColor(0.0,0.0,0.0,0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glLineWidth(1.0);
	glColor3f(0.0,1.0,0.0);
	glBegin(GL_LINES);
	glVertex2f(0.0,0.0);
	glVertex2f(sin(degree*M_PI/180)*.95, cos(degree*M_PI/180)*.95);
	glEnd();

	glFlush();
}

void update_lines(int te){
	degree++;

	glutPostRedisplay();
	glutTimerFunc(10,update_lines,1);
}

int main(int argc,char **argv){
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_SINGLE|GLUT_RGBA);
	glutInitWindowPosition(500,500);
	glutInitWindowSize(400,400);
	glutCreateWindow("Hellooooo");

	glutDisplayFunc(display);
	glutTimerFunc(10,update_lines,1);

	glutMainLoop();
}
