/*	Author		: Chris Skane
	Created		: 20200206
	Modified	: 20200207
	Desc: Bind to a port and launch a thread per-connection, up to a max, which
			allows the program to be interactive without blocking all clients.
	To try: In the program or a copy, fork a new process, replace stdin, stdout
				and possibly stderr with a bi-directional network socket.
				Client then has a direct connection to a process
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

#define PORT 2000
#define MAX_CLIENTS 10

static int server_sock = 0;			// Server file descriptor
static int clients[MAX_CLIENTS];			// Store client file descriptors
static pthread_t threads[MAX_CLIENTS];		// Store thread references
static pthread_barrier_t sock_barrier;		// barrier

void interrupt(int err){
	fprintf(stderr, "Ctrl-C caught. Stopping..\n");

	for(int i = 0;i < MAX_CLIENTS;i++) close(clients[i]);
	close(server_sock);

	exit(0);
}

// Takes client file descriptor
void * prompt_thread(void *client){
	if(!client) return NULL;

	int fd = *(int *)client;
	pthread_barrier_wait(&sock_barrier);
	//printf("FD: %d\n", fd);

	// Whatever else goes here
	const size_t bufLen = 63;
	char buffer[bufLen+1];
	char *namePrompt = "What is your name: ";

	int ret = write(fd, namePrompt, strlen(namePrompt));
	if(ret < 0){
		perror("Could not write to client");
		close(fd);
		return NULL;
	}

	ret = read(fd, buffer, bufLen);
	if(ret < 0){
		perror("Failed to get client response");
		close(fd);
		return NULL;
	}
	buffer[ret] = 0;

	printf("Client message: '%s'\n", buffer);

	// Cleanup stuff
	pthread_t self = pthread_self();
	close(fd);
	for(int i = 0;i < MAX_CLIENTS;i++){
		if(threads[i] == self){
			threads[i] = 0;
			clients[i] = 0;
		}
	}

	return NULL;
}

int main(){
	signal(SIGINT, interrupt);

	char *greeting = "Welcome to my test chamber!\n";

	for(int i = 0;i < MAX_CLIENTS;i++) threads[i] = 0;

	struct sockaddr_in server;
	server.sin_family = AF_INET;
	server.sin_port = htons(PORT);
	server.sin_addr.s_addr = inet_addr("127.0.0.1");

	server_sock = socket(AF_INET, SOCK_STREAM, 0);
	if(server_sock < 0){
		perror("Socket failed to open");
		return server_sock;
	}

	if(bind(server_sock, (struct sockaddr *)&server, sizeof(server))){
		perror("Could not bind to specified port/address");

		close(server_sock);
		return -1;
	}

	if(listen(server_sock, 2)){
		perror("Failed to listen");

		close(server_sock);
		return -2;
	}

	struct sockaddr_in client; // Get client info (Can be used to check duplicates)
	int client_size = sizeof(client);
	int client_sock = 0;
	int slot = 0;

	pthread_barrier_init(&sock_barrier, NULL, 2); // Barrier for thread to get fd
	while(1){
		client_sock = accept(server_sock, (struct sockaddr *)&client, &client_size);
		if(client_sock < 0){
			perror("Could not accept connection");

			if(errno != EAGAIN || errno != EWOULDBLOCK) break;
			continue;
		}

		printf("Accepted client: %s\n", inet_ntoa(client.sin_addr));
		write(client_sock, greeting, strlen(greeting));

		// Find open thread/fd slot, then launch thread
		slot = -1;
		for(int i = 0;i < MAX_CLIENTS;i++) if(!threads[i]) slot = i;
		if(slot >= 0){
			pthread_create(&threads[slot], NULL, prompt_thread, &client_sock);
			pthread_barrier_wait(&sock_barrier); // Make sure thread fd is not changed
			clients[slot] = client_sock;
		}else{
			fprintf(stderr, "Max clients reached. Denying %s\n", inet_ntoa(client.sin_addr));
			close(client_sock);
			client_sock = 0;
		}

		// Standard single-client prompt
		//prompt(client_sock);

		//close(client_sock);
	}

	close(server_sock);

	return 0;
}
