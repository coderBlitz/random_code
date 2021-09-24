#include"png.h"

#define HEADER_CONST 0x0A1A0A0D474E5089

#define REV32(n) ((n << 24) | (((n>>16)<<24)>>16) | (((n<<16)>>24)<<16) | (n>>24))

int parseFile(char *filename){
	if(filename == NULL) return -1;

	/* Parse header
	*/
	FILE *fp = fopen(filename, "rb");
	if(fp == NULL){
		fprintf(stderr, "Cannot open '%s': %s\n", filename, strerror(errno));
		return -1;
	}

	const size_t buffer_size = 1024;
	uint8_t buffer[buffer_size];
	fread(buffer, 1, buffer_size, fp); // Read header
	unsigned long header = *(unsigned long *)buffer;
	printf("Header: 0x%016lX\n", header);

	if(header != HEADER_CONST){
		fprintf(stderr, "Corrupt header.\n");
		return -1;
	}

	uint8_t *chunk_start = buffer + 8;
	uint32_t chunk_length;
	char *chunk_type;
	size_t res = 1;

	while(res){
		chunk_length = *(uint32_t *) chunk_start;
		chunk_length = REV32(chunk_length);
		chunk_type = chunk_start + 4;

		printf("Chunk length: %u\n", chunk_length);
		printf("Chunk type: %.4s\n", chunk_type);

		if(strncmp(chunk_type, "IEND", 4) == 0){
			printf("END reached\n");
			break;
		}

		size_t point = chunk_start + chunk_length + 4 + 8 - buffer;
		if(point < buffer_size){
			chunk_start += chunk_length + 12; // 4 for CRC
		}else{
			printf("Reading..\n");
			while(point > buffer_size){
				res = fread(buffer, 1, buffer_size, fp);
				point -= buffer_size;
			}
			chunk_start = buffer + point;
		}
	}

	fclose(fp);

	return 0;
}
