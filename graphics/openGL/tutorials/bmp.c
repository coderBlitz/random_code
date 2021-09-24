#include<errno.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"bmp.h"

struct _bmp_file readBMP(const char *path){
	char buffer[54];
	struct _bmp_header header;
	struct _bmp_file file;

	FILE *fp = fopen(path, "rb");
	if(fp == NULL){
		fprintf(stderr, "Could not open BMP file '%s': %s\n", path, strerror(errno));
		return file;
	}

	size_t ret = fread(buffer, 1, 54, fp);
	if(ret != 54){
		fprintf(stderr, "Could not read full header\n");
		return file;
	}

	if(buffer[0] != 'B' || buffer[1] != 'M'){
		fprintf(stderr, "Not a valid BMP file\n");
		return file;
	}

	header.magic_1 = buffer[0];
	header.magic_2 = buffer[1];

	header.filesize = *(uint32_t *)&buffer[0x02];
	header.reserved_1 = *(uint16_t *)&buffer[0x06];
	header.reserved_2 = *(uint16_t *)&buffer[0x08];
	header.data_offset = *(uint32_t *)&buffer[0x0A];

	printf("Filesize: %u\n", header.filesize);
	printf("Data pos: %u\n", header.data_offset);

	struct _bmp_dib dib;

	dib.size = *(uint32_t *)&buffer[0x0E];
	dib.width = *(int32_t *)&buffer[0x12];
	dib.height = *(int32_t *)&buffer[0x16];
	dib.color_planes = *(int16_t *)&buffer[0x1A];
	dib.color_depth = *(int16_t *)&buffer[0x1C];
	dib.compression = *(int32_t *)&buffer[0x1E];
	dib.image_size = *(uint32_t *)&buffer[0x22];
	dib.x_resolution = *(int32_t *)&buffer[0x26];
	dib.y_resolution = *(int32_t *)&buffer[0x2A];
	dib.palette_size = *(uint32_t *)&buffer[0x2E];
	dib.important_colors = *(uint32_t *)&buffer[0x32];

	printf("Width: %d\n", dib.width);
	printf("Height: %d\n", dib.height);
	printf("Depth: %d\n", dib.color_depth);
	printf("Compression: %d\n", dib.compression);
	printf("Image size: %u vs %u\n", dib.image_size, dib.width*dib.height*dib.color_depth/8);

	uint32_t image_size = dib.width * dib.height * dib.color_depth/8;

	uint8_t *data = malloc(image_size * sizeof(*data));
	if(data == NULL){
		fprintf(stderr, "Could not allocate data\n");
		return file;
	}

	ret = fread(data, sizeof(*data), image_size, fp);
	if(ret != image_size){
		fprintf(stderr, "Full image was not read.\n");
	}

	file.header = header;
	file.dib = dib;
	file.data = data;

	fclose(fp);
	return file;
}
