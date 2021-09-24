/*	Author		: Chris Skane
	Created		: 20200521
	Modified	: 20200521
	Desc: Bind all ports as a defensive tactic (a poor one at that).
*/

#include<errno.h>
#include<pthread.h>
#include<signal.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>

#include<sys/socket.h>
#include<arpa/inet.h>
#include<netinet/in.h>

#define PORT_MIN 1024
#define PORT_MAX 65535

static int *sockets;			// Server file descriptor
static int socket_count = 0;

void close_all(){
	if(sockets == NULL) return;
	for(int i = 0;i < socket_count;i++) close(sockets[i]);
}

void interrupt(int err){
	fprintf(stderr, "Ctrl-C caught. Stopping..\n");

	close_all();

	exit(0);
}

int main(){
	signal(SIGINT, interrupt);

	// Allocate all sockets
	const int RANGE = PORT_MAX - PORT_MIN + 1;
	sockets = malloc(RANGE * sizeof(*sockets));
	for(int i = 0;i < RANGE;i++) sockets[i] = -1;

	struct sockaddr_in server;
	server.sin_family = AF_INET;
	server.sin_addr.s_addr = inet_addr("127.0.0.1");

	for(int i = PORT_MIN;i <= PORT_MAX;++i){
		int idx = i - PORT_MIN;
		server.sin_port = htons(i);

		sockets[idx] = socket(AF_INET, SOCK_STREAM, 0);
		if(sockets[idx] < 0){
			perror("Socket failed to open");
			continue;
		}
		++socket_count;

		if(bind(sockets[idx], (struct sockaddr *)&server, sizeof(server))){
			fprintf(stderr, "Could not bind to port %hd: %s\n", i, strerror(errno));

			close(sockets[idx]);
			--socket_count;
		}
	}

	while(1){
		sleep(1);
	}

	close_all();

	return 0;
}
