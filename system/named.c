#include<errno.h>
#include<fcntl.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

int main(int argc, char *argv[]){
	printf("Opening..\n");
	int fd = open("named.fifo", O_RDONLY);
	if(fd == -1){
		perror("Open failed");
		exit(1);
	}

	printf("Reading..\n");
	char buffer[1024];
	int res = read(fd, buffer, 1023);
	if(res < 0){
		perror("Read failed");
		exit(1);
	}

	printf("Read:\n%s", buffer);

	printf("Closing..\n");
	close(fd);
}
