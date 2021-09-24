#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"plot.h"

static struct _plot_pixel *colors;

void cleanup(){
	if(colors != NULL){
		free(colors);
	}
}

int main(int argc, char *argv[]){
	glutInit(&argc, argv);
	atexit(&cleanup);

	int X = 400;
	int Y = 400;

	colors = malloc(X*Y * sizeof(struct _plot_pixel));
//	printf("Colors: %p\n", colors);
	for(register uint32_t i = 0;i < X*Y;i++){
//		printf("%d: (%hd,%hd,%hd) @ (%lu,%lu)\n", i, colors[i].red, colors[i].green, colors[i].blue, colors[i].x, colors[i].y);

		colors[i].red = (i/(float)X/Y) * 255.0;
		colors[i].green = (i%X / (float)X) * 255.0;
		colors[i].blue = 0;

		colors[i].x = i%X;
		colors[i].y = floor(i/X);
	}

	plotInit(X, Y, 80, 80);
	setData(colors);
	
	plot();
	return 0;
}
