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

	struct sockaddr_in bcastAddr;
	bcastAddr.sin_family = AF_INET;
	bcastAddr.sin_port = htons(PORT);
	bcastAddr.sin_addr.s_addr = htonl(INADDR_ANY); // Bind to any interface

	if(bind(sock, (struct sockaddr *)&bcastAddr, sizeof(bcastAddr)) == -1){
		fprintf(stderr, "Bind failed: '%s'\n", strerror(errno));
		close(sock);
		exit(1);
	}

	struct sockaddr_in src;
	int thing = 0;
	int len = sizeof(src); // Must initialize for recvfrom()
	char *host = NULL;
	for(int i = 0; ;i++){
		recvfrom(sock, &thing, sizeof(thing), 0, (struct sockaddr *)&src, &len); // No src_addr because it is a broadcast
		host = inet_ntoa(src.sin_addr);
		printf("#%d from '%s': %d!\n", i+1, host, thing);
	}

	close(sock);


	return 0;
}
