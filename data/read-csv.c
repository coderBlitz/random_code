#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<unistd.h>
#include<string.h>

int main(int argc, char *argv[]){
	char *default_file = "aquaculture_imports.csv";
	char *filename = NULL;

	// Get file as argument
	if(argc == 2){
		filename = argv[1];
	}else{
		filename = default_file;
	}

	FILE *fp = fopen(filename,"r");
	if(fp == NULL){
		fprintf(stderr, "Could not open '%s'\n", filename);
		return -1;
	}

	// Data storage buffer. 2-D array of strings (3-D char array)
	char ***buf = malloc(sizeof(*buf));
	if(buf == NULL){
		fprintf(stderr, "Could not allocate initial memory");
		return -2;
	}
	buf[0] = NULL;

	uint64_t lines = 0, max_length = 0; // 'max_length' is the length of the longest field
	uint64_t fields = 1, field = 0; // Total fields, and field iterator
	char c, q = 0; // Temp character, and quotation flag

	/***	TODO: Allocate inside counting loop, to avoid needing to store field length or count twice
			Above todos appear to be completed.
	***/

	// Count lines, fields, and longest field
	for(uint64_t j = 0;(c = fgetc(fp)) != EOF;j++){
		if(c == '\n'){
			++lines;

			// Resize line buffer
			buf = realloc(buf, (lines+1) * sizeof(*buf));
			if(buf == NULL){
				fprintf(stderr, "Could not allocate enough memory");
				return -2;
			}

			// Allocate one field, then allocate field space
			if(fields == 1){
				buf[lines-1] = malloc(sizeof(**buf));
			}
			buf[lines-1][field] = malloc((j+1)*sizeof(***buf));
			if(buf[lines-1][field] == NULL){
				fprintf(stderr, "Could not allocate character memory");
				return -3;
			}

			j = -1; // Reset on new line, negative so iteration makes it 0
			q = 0; // End of line means not in quotes (for now)
			field = 0;
		}
		if(q == 1 && c == '"') q = 0; // Toggle for being inside quotations or not
		else if(c == '"') q = 1; // ""

		// If not inside quotes, and on a comma (field separator)
		if(q == 0 && c == ','){
			// If first line, allocate more fields as necessary.
			// Otherwise allocate field count
			if(!lines){
				++fields;
				buf[lines] = realloc(buf[lines], fields * sizeof(**buf));
				if(buf[lines] == NULL){
					fprintf(stderr, "Could not allocate field memory");
					return -2;
				}
			}else if(!field){
				buf[lines] = malloc(fields * sizeof(**buf));
				if(buf[lines] == NULL){
					fprintf(stderr, "Could not allocate field memory");
					return -2;
				}
			}

			if(j > max_length){
				//printf("i = %d @ %d\n",j,lines);
				max_length = j; // TODO: Account for quotes and comma(s)
			}

			// Allocate field buffer
			buf[lines][field] = malloc((j+1)*sizeof(***buf));
			if(buf[lines][field] == NULL){
				fprintf(stderr, "Could not allocate character memory");
				return -3;
			}

			++field;
			j = 0;
		}
	}
	rewind(fp);

	printf("%d lines\tfields = %u\n", lines, fields); // File info

	printf("Begin data\n");
	q = 0;
	int n = 0; // Field position iterator
	for(uint64_t i = 0;i < lines;i++){
		uint64_t id = 0; // Which field we are processing
		n = 0;
		while((c = fgetc(fp)) != '\n' && c != EOF){
			if(q == 0 && c == '"'){ q = 1; continue; } // Toggle for quotes
			else if(c == '"'){ q = 0; continue; }

			// If un-quoted comma (field separator)
			if(q == 0 && c == ','){
				buf[i][id][n] = '\0'; // End field so we can print
				id++;
				n = 0;
			}else{
				buf[i][id][n] = c; // Store thing
				n++;
			}
		}
		//printf("%s\n",buf[i][2]); // Print field #
		//fflush(stdout); // Flush just in case wierd things happen
	}
	fclose(fp); // Close file
	printf("Done\n");

	for(int n = 0;n < lines;n++){
		for(int j = 0;j < fields;j++) free(buf[n][j]); // Free fields
		free(buf[n]);
	}
	free(buf); // Free field buffer
}
