#include<stdio.h>
#include<stdlib.h>
#include<string.h>

extern char **environ;

int main(int argc, char *argv[]){
	int diff = 0;
	for(int i = 0;i < 100;i++){
		if(environ[i] == NULL) break;
		printf("%d: '%s'\n", i, environ[i]);

		diff = strstr(environ[i], "=") - environ[i];
		if(diff == 4) printf("\tMaybe\n");
	}

}
