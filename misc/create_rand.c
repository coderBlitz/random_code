/* Description: Write a test file with random data, following a pattern.
				Read by parse_data.c
*/

#include<error.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define FILENAME "/tmp/data.txt"

int main(){
	srand(time(0));
	uint64_t bytes_wanted = 2147483648; // Will write at least this many bytes to file
	//uint64_t bytes_wanted = 500;

	FILE *fp = fopen(FILENAME, "w");
	if(fp == NULL){
		perror("Could not open file:");
		return -1;
	}

	uint64_t size = 0;
	uint64_t lines = 0;
	char line[81]; // 80 chars + newline
	while(size < bytes_wanted){
		for(int i = 0;i < 80;++i){
			line[i] = rand()%10 + '0';
		}

		line[0] = '[';	// Just picking some random delimeters
		line[10] = '.';
		line[40] = '-';
		line[80] = '\n';

		size += fwrite(line, sizeof(line[0]), 81, fp);
		++lines;
	}

	printf("Final size: %lu bytes\nLine count: %lu\n", size, lines);

	fclose(fp);
}
