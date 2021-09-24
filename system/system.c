#include<stdio.h>
#include<stdlib.h>

int main(){
	FILE *fp = popen("ls","r");
	if(fp == NULL){
		printf("Could not run command!\n");
		exit(1);
	}

	char c;
	while((c = fgetc(fp)) != EOF) printf("%c",c);
	pclose(fp);
}
