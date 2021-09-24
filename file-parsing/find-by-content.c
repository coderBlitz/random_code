#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>

/* Program searches files for given pattern
 currently case-sensitive
*/

int main(int argc, char *argv[]){
	if(argc == 1){
		printf("find-by-content \"PATTERN\" FILE1 FILE2 ...\n");
		exit(0);
	}

	int len=101;
	char wd[len];
	getcwd(wd,100);
	char pat[50];
	strcpy(pat,argv[1]);

//printf("Current working directory: %s\n",wd);

	char tmp[len];
	for(int i=2;i<argc;i++){//Goes through list of arguments
		strcpy(tmp,wd);
		strcat(tmp,"/");
		strcat(tmp,argv[i]);
//printf("Absolute file path: %s\n",tmp);

		FILE *fp = fopen(tmp,"r");

		int columns=0,max=0;
		char c;
		while((c = getc(fp)) != EOF){//Counts columns, to have a max size value
			if(c == '\n'){
				max = (columns > max)?columns:max;
				columns = 0;
			}
		}
		rewind(fp);

		char *line;
		size_t buf = max;
		while((getline(&line,&buf,fp)) != -1){//Goes through each line in file
			if(strstr(line,pat) != NULL){
				printf("%s\n",tmp);
				break;
			}
		}
	}
}
