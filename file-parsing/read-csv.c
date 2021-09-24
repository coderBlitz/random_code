#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>

int main(int argc, char *argv[]){
	FILE *fp = NULL;
	if(argc > 1){
		printf("Opening '%s'\n", argv[1]);
		fp = fopen(argv[1], "r");
	}else{
		fp = fopen("/home/chris/Downloads/aquaculture_imports.csv","r");
	}
	if(fp == NULL){
		fprintf(stderr, "Could not open file.\n");
		exit(1);
	}

	int lines = 0,i = 0;// 'i' is the length of the longest field
	int field_max = 0, fields = 1, is_counting = 1;
	char c,q = 0;
	for(int j = 0;(c = fgetc(fp)) != EOF;j++){// Count lines, # fields, longest field
		if(c == '\n'){
			lines++;
			j = -1;// Reset on new line, negative so iteration makes it 0
			q = 0;// End of line means not in quotes
			is_counting = 0;
		}
		if(q == 1 && c == '"') q = 0;// Toggle for being inside quotations or not
		else if(c == '"') q = 1;// ""

		if(q == 0 && c == ','){// If not inside quotes, and on a comma (field separator)
			if(j-3 > field_max){
//				printf("i = %d @ %d\n",j-3,lines);
				field_max = j - 3;// Minus 3 because of two quotes and end comma
			}
			j = 0; // Reset net field length
			if(is_counting) fields++;
		}
	}

	rewind(fp);
	printf("%d lines\ti = %d\tfields = %d\n",lines,field_max,fields);// Debug info
	char ***buf = malloc(lines * sizeof(char **));// Allocate array size for whole doc
	for(int n = 0;n < lines;n++){
		buf[n] = malloc(fields * sizeof(char *));// Array of the 10 fields
		for(int j = 0;j < fields;j++){
			buf[n][j] = malloc((field_max+1)*sizeof(char));
		}
	}

	printf("Begin\n");

	q = 0;
	int n = 0;
	for(i = 0;i < lines;i++){
		int id = 0;// Which field we are processing
		while((c = fgetc(fp)) != '\n'){
			if(q == 0 && c == '"'){ q = 1; continue; }// Toggle for quotes
			else if(c == '"'){ q = 0; continue; }

			if(q == 0 && c == ','){// If un-quoted comma (field separator)
				buf[i][id][n] = '\0';// End field so we can print
				id++;// Next field
				n = 0;
			}else{
				buf[i][id][n] = c;// Store thing
				n++;// Holds next buffer spot
				if(n > field_max){
					printf("Field length overrun! Skipping..\n");
					buf[i][id][n] = 0;
					continue;
				}
			}
		}
		q = 0; // Just in case single quote hits end of line
//		printf("%s\n",buf[i][2]);// Print field #
		fflush(stdout);// Flush just in case wierd things happen
	}

	printf("Done\n");

	for(int n = 0;n < lines;n++){
		for(int j = 0;j < fields;j++) free(buf[n][j]);// Free fields
		free(buf[n]);
	}
	fclose(fp);// Close file
	free(buf);// Free field buffer
}
