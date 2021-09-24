#ifndef BMP_H_
#define BMP_H_

#include<stdint.h>
#include<GL/glew.h>

struct _color24{
	uint8_t R;
	uint8_t G;
	uint8_t B;
};

struct _bmp_header{
	uint8_t		magic_1;
	uint8_t		magic_2;
	uint32_t	filesize;
	uint16_t	reserved_1;
	uint16_t	reserved_2;
	uint32_t	data_offset;
};

// 40 byte DIB version
// BITMAPINFOHEADER
struct _bmp_dib{
	uint32_t	size;
	int32_t		width;
	int32_t		height;
	int16_t		color_planes;
	uint16_t	color_depth;
	int32_t		compression;
	uint32_t	image_size;
	int32_t		x_resolution;
	int32_t		y_resolution;
	uint32_t	palette_size;
	uint32_t	important_colors;
	
};

struct _bmp_file{
	struct _bmp_header header;
	struct _bmp_dib dib;
	uint8_t *data;
};

struct _bmp_file readBMP(const char *path);

#endif
