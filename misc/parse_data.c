/* Description: Parse a text file that contains regular message patterns.
				Pairs with the file create_rand.c
*/

#include<error.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define FILENAME "/tmp/data.txt"

char toggle = 0;
char *getField(char *line, int field){
	if(field < 0) return NULL;

	switch(field){
		case 0:
			return strtok(line, ".");
			break;
		case 1:
			return strtok(NULL, "-");
			break;
		case 2:
			return strtok(NULL, "\0");
			break;
	}

	return NULL;
}

int main(){
	uint32_t array_size = 0;
	char ***array = NULL;
	char fields = 3;
	uint64_t memory = 0;

	printf("Allocation of storage done\n");
	FILE *fp = fopen(FILENAME, "r");
	if(fp == NULL){
		perror("Could not open file:");
		return -1;
	}

	uint32_t buffer_size = 1025;
	char *buffer = malloc(buffer_size * sizeof(*buffer));
	if(buffer == NULL){
		perror("Could not allocate memory:");
		return -1;
	}

	char c = 0;
	uint32_t lines = 0;
	uint32_t count = 0;
	while((c = fgetc(fp)) != EOF){
		// End current line and reset for next
		if(c == '\n'){
			buffer[count++] = 0;

			array = realloc(array, (++array_size)* sizeof(*array));
			array[lines] = malloc(fields * sizeof(**array));
			memory += fields * sizeof(**array);
			if(array[lines] == NULL){
				perror("Could not allocate memory:");
				return -1;
			}

			for(char i = 0;i < fields;++i){
				char *pos = getField(buffer, i);
				if(pos != NULL){
					int len = strlen(pos);
					array[lines][i] = malloc(len * sizeof(***array));
					memory += len * sizeof(***array);
					if(buffer == NULL){
						perror("Could not allocate memory:");
						return -1;
					}

					strncpy(array[lines][i], pos, len);
				}else printf("NULL field pos\n");
			}


			++lines;
			count = 0;
		}else{
			buffer[count++] = c;
		}
	}
	memory += array_size * sizeof(*array);

	/*for(uint32_t i = 0;i < lines;++i){
		printf("Line %u:\n", i);
		printf("  Field 1: %s\n", array[i][0]);
		printf("  Field 2: %s\n", array[i][1]);
		printf("  Field 3: %s\n", array[i][2]);
	}*/

	printf("%u lines total\n", lines);
	printf("%lu bytes allocated (w/o buffer)\n", memory);

	free(buffer);
	fclose(fp);

	for(uint32_t i = 0;i < lines;++i){
		if(array[i] != NULL){
			for(int j = 0;j < fields;++j) if(array[i][j] != NULL) free(array[i][j]);
		}
		free(array[i]);
	}
	free(array);
	return 0;
}
