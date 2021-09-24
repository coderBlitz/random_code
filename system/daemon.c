#include<signal.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

void intHandler(int var){
	printf("Got the int %d\n", var);

	system("touch /tmp/alive");

	exit(0);
}

int main(int argc, char *argv[]){
	signal(SIGINT, intHandler);

	char daemon = 0;

	// Get args
	for(int i = 1;i < argc;i++){
		if(argv[i][0] != '-') continue;

		if(argv[i][1] == 'd') daemon = 1;
	}

	if(daemon){
		printf("Spawning daemon..\n");

		int pid = fork();
		if(pid < 0){
			fprintf(stderr, "Could not fork damemon.\n");
			return pid;
		}else if(pid){
			printf("Daemon launched. PID: %d\n", pid);
			return 0;
		}
	}

	while(1){
		sleep(2);
	}

	return 0;
}
