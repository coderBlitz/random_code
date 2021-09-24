#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include<math.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<unistd.h>

const uint32_t WIDTH = 500;
const uint32_t HEIGHT = 500;

void timespecsub(struct timespec *a, struct timespec *b, struct timespec *res){
	res->tv_sec = a->tv_sec - b->tv_sec;
	res->tv_nsec = a->tv_nsec - b->tv_nsec;
	if(res->tv_nsec < 0){
		res->tv_nsec += 1000000000;
		res->tv_sec -= 1;
	}
}

int main(int argc, char *argv[]){
	// Setup
	if(!glfwInit()){
		fprintf(stderr, "Failed to init GLFW\n");
		exit(1);
	}

	glfwWindowHint(GLFW_SAMPLES, 4);

	GLFWwindow *window;

	window = glfwCreateWindow(WIDTH, HEIGHT, "Sim", NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Could not create window\n");
		glfwTerminate();
		exit(1);
	}
	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;

	if(glewInit() != GLEW_OK){
		fprintf(stderr, "Glew failed init");
		exit(2);
	}

	glfwSwapInterval(1);

	float ball_x = WIDTH/2.0;
	float ball_y = HEIGHT/2.0;
	float radius = 25.0;
	uint16_t steps = 90;
	double dr = 2*M_PI / steps;
	double rad = 0.0;
	double dt = 0.0; // Holds the difference in time, rather than setting it
	// Next step add elasticity
	//double elastic = 1.0;

	// State of object in seconds and meters (loosely)
	double ds_dx = 200.0;
	double ds_dy = 0.0;
	double dv_dx = 0.0;
	double dv_dy = 500.0;

	double LIMIT = 1.0/120.0;

	struct timespec now, last, diff;
	clock_gettime(CLOCK_REALTIME, &last);
	// Loop
	do{
		clock_gettime(CLOCK_REALTIME, &now);
		timespecsub(&now, &last, &diff);
		dt = diff.tv_sec + ((float)diff.tv_nsec / (float) 1000000000);

		glClearColor(0.5f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

		glLoadIdentity();
		glMatrixMode(GL_PROJECTION);
		glViewport(0, 0, WIDTH, HEIGHT);
		glOrtho(0.0, WIDTH, HEIGHT, 0.0, 100, -100);

//		printf("dt: %lf\n", dt);
		if(dt >= LIMIT){
			// Update state and transform
			ds_dy += dv_dy * dt; // Update velocity
			ds_dx += dv_dx * dt;

			ball_x += ds_dx * dt; // Update position
			ball_y += ds_dy * dt;

			// Deal with wall collision
			if((ball_x + radius) >= WIDTH){
				ds_dx *= -1;
				ball_x += ds_dx * dt;
			}else if((ball_x - radius) <= 0){
				ds_dx *= -1;
				ball_x += ds_dx * dt;
			}
			if((ball_y + radius) >= HEIGHT){
				ds_dy *= -1;
				ball_y += ds_dy * dt;
			}else if((ball_y - radius) <= 0){
				ds_dy *= -1;
				ball_y += ds_dy * dt;
			}

			last = now;
		}

		glBegin(GL_LINES);
			glColor3f(1.0, 1.0, 0.0);
			glVertex2f(WIDTH/2, 0.0);
			glVertex2f(WIDTH/2, HEIGHT);
			glVertex2f(0.0, HEIGHT/2);
			glVertex2f(WIDTH, HEIGHT/2);
		glEnd();

		glBegin(GL_TRIANGLE_FAN);
			glColor3f(1.0, 1.0, 1.0);
			glVertex2f(ball_x, ball_y);

			rad = 0.0; // Otherwise it rotates start by a little bit, making whole circle rotate
			for(uint16_t i = 0;i <= steps;++i, rad += dr){
				glColor3f(1.0, (float)i/(float)steps, 1.0);
				glVertex2f(ball_x + radius*cos(rad), ball_y + radius*sin(rad));
			}
		glEnd();

		glfwPollEvents();
		glfwSwapBuffers(window);
	}while(glfwWindowShouldClose(window) == 0);

	// Clean-up
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
