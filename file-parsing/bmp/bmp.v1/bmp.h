#ifndef BMP_H
#define BMP_H

#include<stdint.h>
#include<stdio.h>

struct _color24{
	uint8_t blue;
	uint8_t green;
	uint8_t red;
};

#define BMP_HEADER_SIZE 14
struct _bmp_header{
	uint8_t		magic_1;		// Should be 'B'
	uint8_t		magic_2;		// Should be 'M'
	uint32_t	filesize;
	uint16_t	reserved_1;	// Just keep 0
	uint16_t	reserved_2;	// ""
	uint32_t	data_offset;	// Offset of pixel data
};

struct _bmp_dib{
	uint32_t	size;		// Size of DIB header (v1 is 12 bytes)
	uint16_t	width;	// Width of BMP in pixels
	uint16_t	height;	// Height of BMP in pixels
	uint16_t	planes;	// # of color planes. Set to 1
	uint16_t	color_depth;	// Core DIB supports 24-bit
};

struct _bmp_file{
	struct _bmp_header	header;
	struct _bmp_dib		dib;
	struct _color24			**pixels; // 2-D array of pixels, starting from bottom left
};

int writeBMP(FILE *, struct _bmp_file *);
int initBMP(struct _bmp_file *, const uint32_t, const uint32_t);
int cleanBMP(struct _bmp_file *);

#endif
