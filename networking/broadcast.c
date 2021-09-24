#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#include<errno.h>

#include<sys/socket.h>
#include<arpa/inet.h>
#include<netinet/in.h>

#define PORT 2000

int main(int argc, char *argv[]){
	const char *BCAST_ADDR = "192.168.0.255";

	int sock = socket(AF_INET, SOCK_DGRAM, 0);
	if(sock == -1){
		fprintf(stderr, "Socket failed: '%s'\n", strerror(errno));
		exit(1);
	}

	const int bcast = 1;
	if(setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &bcast, sizeof(bcast)) == -1){
		fprintf(stderr, "Socket options failed: '%s'\n", strerror(errno));
		close(sock);
		exit(1);
	}

	struct sockaddr_in dest;
	dest.sin_family = AF_INET;
	dest.sin_port = htons(PORT);
	dest.sin_addr.s_addr = inet_addr(BCAST_ADDR);

	// Do stuff
	int thing = 123456;
	for(int i = 0;i < 10;i++){
		printf("Sending data #%d..\n", i+1);
		sendto(sock, &thing, sizeof(thing), 0, (struct sockaddr *)&dest, sizeof(dest));
		sleep(1);
	}

	close(sock);

	return 0;
}
