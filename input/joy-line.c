#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<pthread.h>
#include<math.h>
#include<fcntl.h>
#include<linux/joystick.h>

#include<GL/glut.h>

static int degrees = 0;// degrees for drawn line
static char pressed_F=0;// Two vars for graphic update method
static char pressed_R=0;
static double rot_y=0.0;

static int fd;// File descriptor for gamepad/joystick interface
struct js_event ev;// Holds controller info

int increment=1,held_inc=0,held_dec=0;
void *js_update(){// Second thread deals exclusively with updating controller
	while(read(fd,&ev,sizeof(ev)) > 0){
		if(ev.type == JS_EVENT_BUTTON){// Only check button presses
			if(ev.number == 0 && ev.value == 1) pressed_F =1;// Forward button 'A'
			else if(ev.number == 0 && ev.value == 0) pressed_F =0;
			if(ev.number == 1 && ev.value == 1) pressed_R =1;// Reverse button 'B'
			else if(ev.number == 1 && ev.value == 0) pressed_R =0;

			if(ev.number == 3 && ev.value == 1){// If Y, then increase by 1
				if(held_inc == 0){ increment++; held_inc=1; }
			}else held_inc=0;
			if(ev.number == 2 && ev.value == 1){// If X, then decrease by 1
				if(held_dec == 0){ increment--; held_dec=1; }
			}else held_dec=0;

			if(ev.number == 6 && ev.value == 1){ printf("\n"); exit(0); }
		}else if(ev.type == JS_EVENT_AXIS){
			if(ev.number == 1) rot_y = ev.value/32767 * 90;
		}
		printf("\rIncrement: %3d",increment);// Just some useful info
		fflush(stdout);
	}
}

double factor=0.9;// Length of line percent (1 is full length of window)
void display(){
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0,0.0,0.0,0.0);// Set Background color RBGA

	glLineWidth(2.0);
	glColor3f(0.0,1.0,0.0);// Line coloe
	glBegin(GL_LINES);
	glVertex3f(0,0,0);// Dead center of window
	glVertex3f(sin(degrees * M_PI/180)*factor, cos(degrees * M_PI/180)*factor, 0.0);// Vertex in rectangular coordinates
	glEnd();


	glRotated(rot_y,-1,0,0);

	glutSwapBuffers();
}

void update(int val){// Screen update function
	if(pressed_F == 1) degrees += increment;
	else if(pressed_R == 1) degrees -= increment;

	glutPostRedisplay();
	glutTimerFunc(1,&update,0);// Call update again
}

int main(int argc,char **argv){
	glutInit(&argc,argv);
//	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_CONTINUE_EXECUTION);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowPosition(10,20);// Offset from top left corner of screen
	glutInitWindowSize(600,600);
	glutCreateWindow("Test");// Create and open window
	glutDisplayFunc(&display);

	fd = open("/dev/input/js0",O_RDONLY);// Open gamepad/joystick file
	if(fd == -1){
		fprintf(stderr,"No joystick plugged in\n");
		exit(1);
	}
	pthread_t pth;// This worker thread simply updates controller info continuously
	pthread_create(&pth,NULL,js_update,NULL);// Starts thread

printf("Starting!\n");
	glutTimerFunc(1,&update,0);// Start the reaction!

	glutMainLoop();
printf("Cleaning..\n");
	close(fd);
printf("Done!\n");
}
