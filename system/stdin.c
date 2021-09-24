#include<errno.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<sys/wait.h>

char code[32] = "Hello";
char *const cmd = "/bin/cat";
char *args[] = {cmd, NULL};

int main(void){
	int pid = 0;
	int des[2];
	int sed[2];
	char buff[64] = {0};

	if(pipe(des) == -1){
		perror("Pipes failed");
		exit(1);
	}
	if(pipe(sed) == -1){
		perror("Pipes failed");
		exit(1);
	}

	pid = fork();
	if(pid == -1){
		perror("Fork failed");
		exit(2);
	}else if(pid == 0){
		close(sed[0]); // Close our read side
		close(des[1]); // Close our write side

		if(dup2(des[0], 0) == -1){
			perror("Duplication failed");
			exit(3);
		}
		close(des[0]);

		if(dup2(sed[1], 1) == -1){
			perror("Stdout duplication failed");
			exit(3);
		}
		close(sed[1]);

		if(execve(cmd, args, NULL) == -1){
			perror("Exec failed");
			exit(4);
		}

		//fgets(buff, 63, stdin);
		//printf("Child stdin: '%s'\n", buff);
	}else{
		close(des[0]);// Close our read side
		close(sed[1]);

		printf("Writing to child: '%s'\n", code);
		write(des[1], code, 32);
		close(des[1]);

		char buffer[32];
		read(sed[0], buffer, 32);
		printf("Child: %s\n", buffer);

		sleep(2);

		while(read(sed[0], buffer, 32)){
			printf("Child: %s\n", buffer);
		}

		close(sed[0]);
		waitpid(pid, NULL, 0);
	}

	return 0;
}
