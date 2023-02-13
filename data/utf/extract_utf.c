/*** extract_utf.c -- Print unicodes for all UTF-8 characters
***/

#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>


typedef uint32_t Unicode_char;


int main(int argc, char *argv[]){
	if(argc != 2){
		printf("Usage: extract_utf /path/to/file1\n");
		return 1;
	}

	FILE *fp = fopen(argv[1], "rb");
	if(fp == NULL){
		perror("Could not open file");
		return 1;
	}

	/* Parse 4096 (buffer_size / 2) bytes at a time, which we'll call a chunk.
	*/
	const size_t buffer_size = 8192 + 8;
	uint8_t buffer[buffer_size];
	int chunk_size = buffer_size - 8;
	int data_size = 0;
	size_t read_count = 0;
	size_t total_count = 0;
	int leftover = 0; // Bytes copied from end of buffer to front, before new read

	while((read_count = fread(buffer + leftover, 1, chunk_size - leftover, fp)) > 0){
		total_count += read_count;
		data_size = read_count + leftover;
		leftover = 0;

		// Iterate through buffer checking for UTF headers
		int pos = 0, remain;
		Unicode_char c;
		while((leftover == 0) && (pos < data_size)){
			remain = chunk_size - pos - 1; // Remaining bytes in chunk buffer
			c = 0;

			// UTF magic byte checks
			if((buffer[pos] & 0xF8) == 0xF0){
				// 4-byte UTF character
				if(remain < 3){
					goto copy_leftovers;
				}

				// Verify that next bytes have correct magic
				if((buffer[pos+1] & 0xC0) == 0x80 && (buffer[pos+2] & 0xC0) == 0x80 && (buffer[pos+3] & 0xC0) == 0x80){
					c =
						(buffer[pos]   & 0x07) << 18 |
						(buffer[pos+1] & 0x3F) << 12 |
						(buffer[pos+2] & 0x3F) << 6  |
						(buffer[pos+3] & 0x3F);
				}

				pos += 3;
			}else if((buffer[pos] & 0xF0) == 0xE0){
				// 3-byte UTF character
				if(remain < 2){
					goto copy_leftovers;
				}

				// Verify that next bytes have correct magic
				if((buffer[pos+1] & 0xC0) == 0x80 && (buffer[pos+2] & 0xC0) == 0x80){
					c =
						(buffer[pos]   & 0x0F) << 12 |
						(buffer[pos+1] & 0x3F) << 6 |
						(buffer[pos+2] & 0x3F);
				}

				pos += 2;
			}else if((buffer[pos] & 0xE0) == 0xC0){
				// 2-byte UTF character
				if(remain < 1){
					goto copy_leftovers;
				}

				// Verify that next byte has correct magic
				if((buffer[pos+1] & 0xC0) == 0x80){
					c =
						(buffer[pos]   & 0x1F) << 6 |
						(buffer[pos+1] & 0x3F);
				}

				pos += 1;
			}else if((buffer[pos] & 0x80) == 0x00){
				// 1-byte UTF AKA standard ASCII

				c = buffer[pos] & 0x7F; // Low 7 bits
			}else{
				// High/extended ASCII
			}

			// Print all characters (or just the non-ascii ones)
			//if(c){
			if(c > 0x7F){
				printf("Unicode U+%04X at %lu\n", c, (total_count - read_count) + pos);
			}

			pos += 1;
			continue;

			// For when UTF sequence interrupted by end of chunk
			copy_leftovers:
			leftover = remain + 1;
			memmove(buffer, buffer + pos, leftover);
		}
	}

	fclose(fp);
	fp = NULL;

	// Finish
	printf("total count: %lu\n", total_count);

	if(leftover > 0){
		fprintf(stderr, "Incomplete UTF character at end of file.\n");
	}

	return 0;
}
