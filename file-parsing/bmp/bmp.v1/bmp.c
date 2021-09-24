#include"bmp.h"
#include<stdio.h>
#include<stdlib.h>

int initBMP(struct _bmp_file *file, const uint32_t width, const uint32_t height){
	if(width <= 0 || height <= 0){
		fprintf(stderr, "Cannot initialize BMP file with 0 width/height.\n");
		return -1;
	}

	file->header.magic_1 = 'B';
	file->header.magic_2 = 'M';
	file->header.filesize = 0;
	file->header.reserved_1 = 0;
	file->header.reserved_2 = 0;
	file->header.data_offset = 26;

	file->dib.size = 12;
	file->dib.width = width;
	file->dib.height = height;
	file->dib.planes = 1;
	file->dib.color_depth = 24;

	file->pixels = malloc(height * sizeof(struct _color24 *));
	if(file->pixels == NULL){
		fprintf(stderr, "Could not allocate memory for pixels\n");
		return -1;
	}

	for(uint32_t i = 0;i < height;i++){
		file->pixels[i] = malloc(width * sizeof(struct _color24));

		// If any row fails to allocate, de-allocate everything up to now and quit
		if(file->pixels[i] == NULL){
			fprintf(stderr, "Could not allocate memory for pixel row. Stopping.\n");

			for(uint32_t j = i;j >= 0;j--){
				free(file->pixels[j]);
				file->pixels[j] = NULL;
			}
			free(file->pixels);
			file->pixels = NULL;

			return -1;
		}
	}
}

int cleanBMP(struct _bmp_file *file){
	if(file == NULL) return -1;
	if(file->pixels == NULL) return -1;

	uint32_t height = file->dib.height;
	for(uint32_t i = 0;i < height;i++){
		if(file->pixels[i] != NULL){
			free(file->pixels[i]);
			file->pixels[i] = NULL;
		}else fprintf(stderr, "Pixel row %u was null.\n");
	}
	free(file->pixels);
	file->pixels = NULL;
}

int writeBMP(FILE *fp, struct _bmp_file *file){
	if(fp == NULL){
		fprintf(stderr, "Cannot write to an empty file.\n");
		return -1;
	}else if(file->dib.size <= 0){
		fprintf(stderr, "Cannot write file, invalid DIB size\n");
		return -1;
	}else if(file->pixels == NULL){
		fprintf(stderr, "Pixel data empty, please add at least 1 pixel\n");
		return -1;
	}

	rewind(fp);

	uint32_t width = file->dib.width;
	uint32_t height = file->dib.height;
	uint32_t depth = file->dib.color_depth / 8;
//	printf("Depth: %u\n", depth);

	uint32_t total_header_size = BMP_HEADER_SIZE + file->dib.size;
//	file->header.filesize = total_header_size + width*height*depth + height*((4 - (width*depth)%4)%4);
	file->header.filesize = 0;

	uint32_t wrote = total_header_size;
	// Write the headers
	fwrite(&file->header, BMP_HEADER_SIZE, 1, fp);
	fwrite(&file->dib, file->dib.size, 1, fp);

	uint8_t pad = 0;
	for(uint32_t row = 0;row < height;row++){
		if(file->pixels[row] == NULL){
			fprintf(stderr, "Row %u is null. Continuing..\n");
			continue;
		}
		// Write the color data, as width * color_depth		
		wrote += depth * fwrite(file->pixels[row], depth, width, fp);
/*		for(uint32_t col = 0; col < width; col++){
			fwrite(&file->pixels[row][col], sizeof(struct _color24), 1, fp);
		}*/

		if((width*depth)%4){
//			printf("Padding with %u zero bytes\n", (width*depth)%4);
			wrote += fwrite(&pad, sizeof(pad), 4 - (width*depth)%4, fp);
		}
	}
	printf("Filesize: %u B\n", wrote);

	fseek(fp, 2, SEEK_SET);
	fwrite(&wrote, sizeof(wrote), 1, fp);

	return 0;
}
