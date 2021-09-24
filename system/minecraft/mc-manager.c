#include<errno.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<sys/wait.h>
#include<pthread.h>

#define JAR_FILE "minecraft_server.1.14.4.jar"

struct server_message{
	uint32_t timestamp;
	char source[32];
	char type[8];
	char *content;
};

char *cmd_stop = "stop\n";
char *const cmd = "/bin/java";
/*
java -Xms12000M -Xmx12000M -XX:+UseG1GC -XX:+ParallelRefProcEnabled -XX:+UnlockExperimentalVMOptions -XX:+DisableExplicitGC -XX:-OmitStackTraceInFastThrow -XX:+AlwaysPreTouch -XX:MaxGCPauseMillis=100 -XX:G1NewSizePercent=30 -XX:G1MaxNewSizePercent=40 -XX:G1HeapRegionSize=8M -XX:G1ReservePercent=20 -XX:G1HeapWastePercent=5 -XX:G1MixedGCCountTarget=8 -XX:InitiatingHeapOccupancyPercent=15 -XX:G1MixedGCLiveThresholdPercent=90 -XX:G1RSetUpdatingPauseTimePercent=5 -XX:SurvivorRatio=32 -XX:MaxTenuringThreshold=1 -Dusing.aikars.flags=true -Daikars.new.flags=true -jar tuinity.jar
*/
char *args[] = {cmd, "-server", "-d64",
				"-Xmx12G", "-Xms12G",
				"-XX:+UseConcMarkSweepGC", "-XX:+UseParNewGC", "-XX:ParallelGCThreads=2",
				"-XX:+AggressiveOpts", "-jar", JAR_FILE, "nogui", NULL};

int pid = 0;
char running = 0;
char restarting = 0;

int outputDes[2]; // Used to read stdout
int inputDes[2]; // Used to write stdin

int start_server(){
	printf("Spawning new process..\n");
	if(pipe(inputDes) == -1){
		perror("Pipes failed");
		exit(1);
	}

	if(pipe(outputDes) == -1){
		perror("Pipes failed");
		exit(1);
	}

	pid = fork();
	if(pid == -1){
		perror("restart fork failed");
		exit(2);
	}else if(pid == 0){
		close(inputDes[1]); // Close our write side
		close(outputDes[0]); // Close our read side

		if(dup2(inputDes[0], 0) == -1){
			perror("Duplication failed");
			exit(3);
		}
		close(inputDes[0]);

		if(dup2(outputDes[1], 1) == -1){
			perror("Duplication failed");
			exit(3);
		}
		close(outputDes[1]);

		if(execve(cmd, args, NULL) == -1){
			perror("Exec failed");
			exit(4);
		}
	}
	close(inputDes[0]);// Close our read side
	close(outputDes[1]);// Close oure write side
}

int restart_server(int fd){
	printf("Restarting..\n");
	if(!waitpid(pid, NULL, WNOHANG)) write(fd, cmd_stop, strlen(cmd_stop)); // If not already stopped

	restarting = 1;
	while(!waitpid(pid, NULL, WNOHANG)){
		sleep(1);
	}

	close(inputDes[0]);
	close(inputDes[1]);
	close(outputDes[0]);
	close(outputDes[1]);

	start_server();

	restarting = 0;

	return 0;
}

// Call for each line, to parse information from it
void parse_line(char *line){
	struct server_message msg;
	msg.timestamp = 0;

	char time[9];
	memcpy(time, &line[1], 8); // Get only the time information

	// Parse the timestamp in the message
	char *part = NULL;
	part = strtok(time, ":");
	while(part != NULL){
		uint32_t num = atoi(part);

		msg.timestamp <<= 8;
		msg.timestamp |= num & 0xFF; // Only need first 8 bits

		part = strtok(NULL, ":");
	}

	// Parse the message origin and type
	char source[64];
	memset(source, 0, 64);

	uint32_t len = strlen(line);
	for(uint32_t i = 12; i < 77; ++i){
		if(line[i] == ']'){
			strncpy(source, &line[12], i-12);
			source[i - 11] = 0;
			break;
		}
	}

	// What's left is the content
	uint32_t content_offset = strlen(source) + 15; // 15 = 10 (time) + 2 (space) + 2 (brackets) + 1 (colon)
	char *remain = &line[content_offset];

	//printf("Remain: '%s'\tsource: '%s' (%u)\n", remain, source, strlen(source));

	strcpy(msg.source, strtok(source, "/")); // Do after strlen, since it messes with it
	strcpy(msg.type, strtok(NULL, "/"));

	if(remain[0] == '[' || remain[0] == '<'){
		// Chat message or user content
	}else{
		// Server auto message
		if(strstr(remain, "joined")){
			char *name = strtok(remain, " ");

			char command[128];
			sprintf(command, "title %s subtitle \"Sorry if this is annoying, just testing.\"\n", name);
			write(inputDes[1], command, strlen(command));

			sprintf(command, "title %s title \"Welcome %s\"\n", name, name);
			write(inputDes[1], command, strlen(command));

			printf("'%s' has joined the server\n", name);
			remain[strlen(remain)] = ' '; // So original message is untouched
		}
	}
}

// Run timed commands
void * maintenance(void *thing){
	int *con = (int *)thing;

	char *command = "say Hi\n";
	uint32_t i = 0;
	while(running){
		if(waitpid(pid, NULL, WNOHANG)){
			printf("Server must have crashed, restarting..\n");
			restart_server(con[1]);
		}
		//write(con[1], command, strlen(command));
		if(i == 20){
			//restart_server(con[1]); // Purely to test the restart mechanism (more likely every 1 day)
		}

		sleep(1);
		++i;
	}
}

// Get keyboard inputs
void * watch(void *thing){
	char c = 0;
	int count = 0;
	char line[256];
	int *con = (int *)thing;

	while((c = getc(stdin))){
		if(c == '\n'){
			if(restarting){
				printf("Ignoring all commands while restarting\n");
				continue;
			}

			line[count++] = c;
			line[count] = 0;
			//printf("%s\n", line);
			if(!waitpid(pid, NULL, WNOHANG)){
				write(con[1], line, count);
			}else{
				printf("Server died, can't write\n");
			}
			count = 0;

			if(!strcmp(line, "stop\n")){
				running = 0;
				break;
			}
		}else line[count++] = c;
	}
	sleep(1);
	close(con[1]);

	printf("Keyboard thread done\n");
	pthread_exit(0);
}

int main(void){
	char buff[64] = {0};

	start_server();

	pthread_t input;
	if(pthread_create(&input, NULL, &watch, inputDes)){
		printf("Failed to thread\n");
	}

	running = 1;
	pthread_t timed;
	if(pthread_create(&timed, NULL, &maintenance, inputDes)){
		printf("Failed to thread\n");
	}

	uint32_t N = 1024;
	char buffer[N];
	char line[N];
	int last = 0;
	int r = 0;
	while((r = read(outputDes[0], buffer, N)) || running){
		if(restarting){
			sleep(1);
		}
		last = 0;
		for(int i = 0;i < r;++i){
			if(buffer[i] == '\n'){
				//printf("i: %d\tlast: %d\n", i, last);
				strncpy(line, &buffer[last], i-last);
				line[i-last] = 0;
				parse_line(line);
				printf("%s\n", line);

				last = i+1;
			}
		}
	}

	close(inputDes[1]);

	pthread_cancel(input);
	pthread_cancel(timed);
	pthread_join(input);
	pthread_join(timed);

	waitpid(pid, NULL, 0);
	printf("Done\n");

	close(outputDes[0]);

	printf("\n");

	return 0;
}
