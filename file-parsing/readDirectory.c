#include<stdio.h>
#include<stdlib.h>
#include<dirent.h>

int main(int argc, char *argv[]){
	char *dirName = "/home/chris";// Directory to be opened
	if(argc == 2) dirName = argv[1];

	DIR *direct = opendir(dirName);// Open directory
	if(direct == NULL){
		printf("Could not open directory \"%s\"\n",dirName);
		exit(1);
	}

	struct dirent *dirptr = NULL;// Pointer to store filenames in Directory

	printf("Files in \"%s\":\n",dirName);
	for(int n=1;(dirptr = readdir(direct)) != NULL;n++){// While there are still files,
		printf("%d: %s\n",n,dirptr->d_name);// Print the file name 'd_name' from dirptr struct
	}

	closedir(direct);// Close the directory
}
