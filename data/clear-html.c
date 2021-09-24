#include<stdio.h>
#include<stdlib.h>

int main(int argc, char **argv){
	if(argc < 2){
		printf("No input file given.\n");
		exit(1);
	}

	FILE *fp = fopen(argv[1],"r");
	if(fp == NULL){
		printf("Could not open file\n");
		exit(2);
	}

	char c = 0, tag = 0, last = 0, newlines = 0;
	while((c = fgetc(fp)) != EOF){
//		printf("C = (%d)\tLast = (%d)\n",c,last);
		if(c == '<'){
			tag = 1;// We don't need to print tags either way
			continue;
		}
		else if(c == '>'){
			tag = 0;
			continue;
		}

		if(!tag && (c != '\n' || last != '\n')){
			last = c;
			if(c == '\n'){
				newlines = (newlines+1)%6;

				if(newlines%6 == 0) printf("");// At the end of each row
				else{
					printf(",");
					continue;
				}
			}
			printf("%c",c);
		}

	}
	printf("\n");

	fclose(fp);
}
