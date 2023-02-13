#include<stdio.h>
#include<stdlib.h>
#include"png.h"

int main(int argc, char *argv[]){
	char *filename = "screenshot_lg.png";

	if(argc == 2){
		filename = argv[1];
	}

	parseFile(filename);

	size_t len = 4;
	uint8_t dat[] = {0x00,0x00,0x00,0x00};
	uint32_t crc = CRC32(0, dat, len * sizeof(*dat));
	printf("CRC = 0x%08X\n", crc);

	return 0;
}
