#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include"bmp.h"

int main(int argc, char *argv[]){
	srand(time(0));
	struct _bmp_file test;

	const int width = 765;
	const int height = 745;
	initBMP(&test, width, height);

	struct _color24 black = {0,0,0};
	struct _color24 white = {255,255,255};
	struct _color24 red = {0,0,255};
	struct _color24 yellow = {0, 255, 255};
	struct _color24 green = {0,255,0};
	struct _color24 teal = {255,255,0};
	struct _color24 blue = {255,0,0};
	struct _color24 purple = {255,0,255};
	struct _color24 gray = {128,128,128};

	for(uint32_t row = 0; row < height; row++){
		for(uint32_t col = 0; col < width; col++){
//			test.pixels[row][col].red = 0;
//			test.pixels[row][col].green = ((float)row / (float)height) * 255;
//			test.pixels[row][col].blue = rand() % 255;
			test.pixels[row][col] = yellow;
		}
	}

/*	test.pixels[0][0] = white;
	test.pixels[0][1] = red;
	test.pixels[0][2] = yellow;
	test.pixels[1][0] = green;
	test.pixels[1][1] = teal;
	test.pixels[1][2] = blue;
	test.pixels[2][0] = white;
	test.pixels[2][1] = black;
	test.pixels[2][2] = white;;
*/

	FILE *fp = fopen("out.bmp", "wb");
	if(fp == NULL){
		fprintf(stderr, "Could not open file\n");
		return 1;
	}

	writeBMP(fp, &test);
//	fwrite(&test.header, sizeof(test.header), 1, fp);

	fclose(fp);
	cleanBMP(&test);
}
