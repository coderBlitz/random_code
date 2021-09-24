#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>

int main(){
	FILE *fp = fopen("aquaculture_imports.csv","r");

	int lines=0,i=0;// 'i' is the length of the longest field
	char c,q=0;
	for(int j=0;(c = fgetc(fp)) != EOF;j++){// Count lines and the longest field
		if(c == '\n'){
			lines++;
			j = -1;// Reset on new line, negative so iteration makes it 0
			q=0;// End of line means not in quotes (for now)
		}
		if(q == 1 && c == '"') q=0;// Toggle for being inside quotations or not
		else if(c == '"') q=1;// ""

		if(q == 0 && c == ','){// If not inside quotes, and on a comma (field separator)
			if(j-3 > i){
//				printf("i = %d @ %d\n",j-3,lines);
				i=j - 3;// Minus 3 because of two quotes and end comma
			}
			j=0;
		}
	}
	rewind(fp);
	printf("%d lines\ni = %d\n",lines,i);// Debug info
	char ***buf = malloc(lines * sizeof(char **));// Allocate array size for whole doc
	for(int n=0;n<lines;n++){
		buf[n] = malloc(10*sizeof(char *));// Array of the 10 fields
		for(int j=0;j<10;j++) buf[n][j] = malloc((i+1)*sizeof(char));// Allocate each field of max length, plus 1 for \0
	}

printf("BEgin\n");
	q=0;
	int n=0;
	for(i=0;i<lines;i++){
		int id=0;// Which field we are processing
		while((c = fgetc(fp)) != '\n'){
			if(q == 0 && c == '"'){ q=1; continue; }// Toggle for quotes
			else if(c == '"'){ q=0; continue; }

			if(q == 0 && c == ','){// If un-quoted comma (field separator)
				buf[i][id][n] = '\0';// End field so we can print
				id++;// Next field
				n=0;
			}else{
				buf[i][id][n] = c;// Store thing
				n++;// Holds next buffer spot
			}
			}
		printf("%s\n",buf[i][2]);// Print field #
		fflush(stdout);// Flush just in case wierd things happen
	}

printf("Done\n");
	for(int n=0;n<lines;n++){
		for(int j=0;j<10;j++) free(buf[n][j]);// Free 10 fields
		free(buf[n]);
	}
	free(buf);// Free field buffer
	fclose(fp);// Close file
}
