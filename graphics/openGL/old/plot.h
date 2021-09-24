/* REQUIRED LIBRARIES:
	GL
	glut
	GLU
*/
#ifndef PLOT_H
#define PLOT_H

#include<GL/glut.h>

struct _plot_pixel{
	int16_t red;
	int16_t green;
	int16_t blue;
	uint64_t x;
	uint64_t y;
};

int plot();
void plotInit(int, int, int, int); // width, height, x, y
static void display();
void setData(struct _plot_pixel *);

#endif
