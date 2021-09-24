#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include"bmp.h"

const struct _color24 black = {0,0,0};
const struct _color24 white = {255,255,255};
const struct _color24 red = {0,0,255};
const struct _color24 yellow = {0, 255, 255};
const struct _color24 green = {0,255,0};
const struct _color24 teal = {255,255,0};
const struct _color24 blue = {255,0,0};
const struct _color24 purple = {255,0,255};
const struct _color24 gray = {128,128,128};

int main(int argc, char *argv[]){
	srand(time(0));

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

	struct _bmp_file file;
	readBMP("bee.bmp");

	return 0;
}
