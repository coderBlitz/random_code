#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

int main(int argc,char **argv){
	if(argc == 1) printf("Hello there\n");
	else printf("Hello %s\n",argv[1]);
}
