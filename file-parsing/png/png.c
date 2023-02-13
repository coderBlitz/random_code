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
	size_t length = 0;

	while(res){
		chunk_length = *(uint32_t *) chunk_start;
		chunk_length = REV32(chunk_length);
		chunk_type = chunk_start + 4;
		printf("Chunk '%.4s': %8lu - %8lu\n", chunk_type, chunk_start - buffer + length, length + chunk_length + 4);

		//printf("Chunk length: %u\n", chunk_length);
		//printf("Chunk type: %.4s\n", chunk_type);

		if(strncmp(chunk_type, "IEND", 4) == 0){
			printf("END reached\n");
			break;
		}

		size_t point = chunk_start + chunk_length + 4 + 8 - buffer;
		if(point < buffer_size){
			chunk_start += chunk_length + 12; // 4 for length, 4 for type, 4 for CRC
		}else{
			// TODO: Just use fseek instead of loop?
			printf("Reading..\n");
			while(point > buffer_size){
				res = fread(buffer, 1, buffer_size, fp);
				length += res;
				point -= buffer_size;
			}
			chunk_start = buffer + point;
		}
	}

	fclose(fp);

	return 0;
}

size_t PNGcreateChunk(const char *type, const void *data, size_t length, void *chunk_out){
	uint8_t *out = chunk_out;

	*(uint32_t *)chunk_out = REV32(length); // Copy length to chunk
	memcpy(out + 4, type, 4); // Copy chunk type
	memcpy(out + 8, data, length); // Copy chunk data
	uint32_t crc = CRC32(0, out + 4, length + 4);
	printf("CRC = 0x%X\n", crc);
	crc = REV32(crc);
	memcpy(out + 8 + length, &crc, 4);

	return length + 12;
}

uint32_t CRC32(uint32_t crc, const char *buf, const size_t len){
	static uint32_t table[256];
	static int have_table = 0;
	uint32_t rem;
	uint8_t octet;
	int i, j;
	const char *p, *q;
 
	/* This check is not thread safe; there is no mutex. */
	if (have_table == 0) {
		/* Calculate CRC table. */
		for (i = 0; i < 256; i++) {
			rem = i;  /* remainder from polynomial division */
			for (j = 0; j < 8; j++) {
				if (rem & 1) {
					rem >>= 1;
					rem ^= 0xedb88320;
				} else
					rem >>= 1;
			}
			table[i] = rem;
		}
		have_table = 1;
	}
 
	crc = ~crc;
	q = buf + len;
	for (p = buf; p < q; p++) {
		octet = *p;  /* Cast to unsigned octet. */
		crc = (crc >> 8) ^ table[(crc & 0xff) ^ octet];
	}
	return ~crc;
}
