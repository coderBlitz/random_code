#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<sys/mman.h> // For mmap
#include<string.h>
#include<fcntl.h>

int main(){
	char *test, *fileMap;
	int file = 0;

	file = open("Libraries.txt",O_RDONLY);
	printf("file = %d\n",file);
	if(file < 0) exit(1);


	fileMap = mmap(NULL, 512, PROT_READ, MAP_PRIVATE, file, 0);
	test = mmap(NULL, 1024, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);

	strcpy(test,"Hello world!");

	printf("Test map: %s\n",test);
	printf("File contents follow:\n%s------ FILE CONTENTS END HERE -------\n",fileMap);


	munmap(test,1024);
	munmap(fileMap, 512);
	close(file);
}
