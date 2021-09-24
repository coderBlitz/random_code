#include<GL/glut.h>

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<time.h>
#include<math.h>

#define num_threads 50// Define number of threads

static int degree[num_threads];
void display(){
	glClearColor(0.0,0.0,0.0,0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glLineWidth(1.0);
	glColor3f(0.0,1.0,0.0);
	glBegin(GL_LINES);

	for(int i=0;i<num_threads;i++){
		glVertex2f(0.0,0.0);
		glVertex2f(sin(degree[i]*M_PI/180)*.95, cos(degree[i]*M_PI/180)*.95);
	}
	glEnd();

	glFlush();
}

static int timings[num_threads];
void update(int timerID){
	degree[timerID] += 1;

	glutPostRedisplay();
	glutTimerFunc(timings[timerID],update,timerID);
}

int main(int argc,char **argv){
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_SINGLE|GLUT_RGBA);
	glutInitWindowPosition(200,100);
	glutInitWindowSize(1000,1000);
	glutCreateWindow("So many circles!");
	glutDisplayFunc(display);

	srand(time(0));
	for(int i=0;i<num_threads;i++) degree[i] = 0;
	for(int i=0;i<num_threads;i++){
		timings[i] = i+1;
		glutTimerFunc(timings[i],update,i);// Allows for different timing
	}

	glutMainLoop();
}
