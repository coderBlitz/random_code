#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<fcntl.h>

int main(){
	int fd[2];
	pipe(fd);

	int N = 33;
	char buffer[N];
	for(int i = 0;i < N;i++) buffer[i] = 0;

	pid_t child = fork();
	if(child < 0){
		fprintf(stderr, "Could not fork");
		exit(1);
	}

	if(child){
		snprintf(buffer, N-1, "Hello World");

		write(fd[1], buffer, N-1);
	}else{
		read(fd[0], buffer, N-1);
		printf("Child recieved: %.*s\n", N-1, buffer);
	}
}
