#include<errno.h>
#include<fcntl.h>
#include<signal.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<unistd.h>
#include<arpa/inet.h>
#include<netinet/in.h>
#include<sys/epoll.h>
#include<sys/socket.h>

/**
TODO: Implement a state-tracking system for each connection (what action to perform next)
**/

short SERVER_PORT = 2173;
long MAX_CONNECTIONS = 4; // Connection state buffer length
long MAX_EPOLL_EVENTS = 128;
double conn_timeout = 10.0; // 10 seconds for connection to live
//struct timespec epoll_timeout = {0, 1000000000}; // Epoll timeout 1 sec (TODO: Use if glibc >= 2.35)
int epoll_timeout = 500; // 1 sec == 1000 ms

struct conn_state {
	unsigned char active; // Non-zero (true) if element is in use, zero otherwise (no active connection)
	struct timespec last_access; // Last time connection was accessed (data was sent/received)
	int fd; // Connection socket
	struct sockaddr_in address;
	size_t data_length; // How much data is currently in buffer
	size_t buffer_size; // Allocated buffer size (is constant currently, but allows for dynamic sizing later)
	unsigned char buffer[2048]; // Data that was received already (can make normal pointer if desired)
};


/** Functions
**/
// Return difference between two timespecs
double timespecdiff(struct timespec *a, struct timespec *b){
	return (a->tv_sec - b->tv_sec) + (a->tv_nsec - b->tv_nsec) / 1000000000.0;
}

// Interact with the client fd's
int handle_client(struct epoll_event *ev){
	struct conn_state *client = (struct conn_state *) ev->data.ptr;

	int addr = ntohl(client->address.sin_addr.s_addr);
	unsigned short port = ntohs(client->address.sin_port);
	//printf("Handling client: %d.%d.%d.%d:%hu\n", (addr >> 24) & 0xFF, (addr >> 16) & 0xFF, (addr >> 8) & 0xFF, addr & 0xFF, port);
	//printf("client event: 0x%X\n", ev->events);
	/*printf("POLLIN = %d\n", EPOLLIN);
	printf("POLLPRI = %d\n", EPOLLPRI);
	printf("POLLOUT = %d\n", EPOLLOUT);
	printf("POLLERR = %d\n", EPOLLERR);
	printf("POLLHUP = %d\n", EPOLLHUP);
	printf("POLLRDHUP = %d\n", EPOLLRDHUP);*/

	clock_gettime(CLOCK_REALTIME, &client->last_access);

	// Ret should be returned, res is scratch.
	int ret = 0, res = 0;

	// Events that can happen when pipe closed (client triggered)
	if(ev->events & (EPOLLHUP | EPOLLERR))
		return 1;

	//close(client->fd);
	//client->active = 0;
	//return 1; // Always close for now

	// If data sent, read into buffer then send back
	if(ev->events & EPOLLIN){
		res = read(client->fd, client->buffer, client->buffer_size);
		if(res < 0){
			perror("Read from client failed");

			return ret;
		}

		client->data_length = res;
		res = send(client->fd, client->buffer, client->data_length, MSG_NOSIGNAL);
		if(res < 0){
			perror("Send to client failed");
		}
	}

	return ret;
}

// Doesn't need to do anything besides exist
void handler(int code){}

int main(int argc, char *argv[]){
	int ret;

	// Init epoll instance
	int epoll_fd = epoll_create1(EPOLL_CLOEXEC);
	if(epoll_fd < 0){
		perror("Epoll initialize failed");
		return errno;
	}

	// Set handler on sigint so epoll triggers EINTR on Ctrl-c
	struct sigaction epoll_sigs;
	epoll_sigs.sa_handler = &handler;
	sigemptyset(&epoll_sigs.sa_mask);
	epoll_sigs.sa_flags = 0;
	struct sigaction old;
	sigaction(SIGINT, &epoll_sigs, &old);

	/* Do networking setup
	TODO: Add flag to choose TCP or UDP (changes init calls, and send/recv calls)
	*/
	int server_sock = socket(AF_INET, SOCK_STREAM, 0); // TCP server (currently)
	if(server_sock < 0){
		perror("Socket creation failed");
		return errno;
	}

	// Make non-blocking socket
	if(fcntl(server_sock, F_SETFL, O_NONBLOCK) < 0){
		perror("Failed to set non-blocking");
		return errno;
	}

	// Bind
	socklen_t in_size = sizeof(struct sockaddr_in);
	struct sockaddr_in local_addr;
	local_addr.sin_family = AF_INET;
	local_addr.sin_port = htons(SERVER_PORT);
	local_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	if(bind(server_sock, (struct sockaddr *) &local_addr, in_size) < 0){
		perror("Bind failed");
		return -1;
	}

	// Listen (queue double the allowed capacity)
	if(listen(server_sock, 2 * MAX_CONNECTIONS)){
		perror("Listen failed");
		return -1;
	}

	/* Main event loop
	TODO: Potentially add server socket fd to epoll to listen for incoming connections
		User data could be pointer to conn_state if connection, NULL for server socket
	*/
	// Initialize state array
	long num_conns = 0;
	struct conn_state connections[MAX_CONNECTIONS];
	for(long i = 0;i < MAX_CONNECTIONS;i++){
		connections[i].active = 0;
	}

	// Epoll struct for adding client sockets
	struct epoll_event client_conn;
	client_conn.events = EPOLLIN; // Verify output not needed

	struct epoll_event conn_events[MAX_EPOLL_EVENTS]; // Allow up to max_connection events at once (for now)

	long next_open = 0;

	int scratch_fd, num_events;
	struct timespec current_time;
	while(1){
		// Accept all incoming connections (or as many as possible), make unblocking connections
		while(num_conns < MAX_CONNECTIONS){
			// Get next connection
			in_size = sizeof(struct sockaddr_in);
			scratch_fd = accept(server_sock, (struct sockaddr *) &connections[next_open].address, &in_size);
			if(scratch_fd < 0)
				break;
			printf("New con\n");

			client_conn.data.ptr = connections + next_open; // Pointer to state

			// Make non-blocking socket
			if(fcntl(scratch_fd, F_SETFL, O_NONBLOCK) < 0){
				perror("Failed to set client non-blocking");
				close(scratch_fd);
				continue;
			}

			// Add fd to EPOLL
			errno = 0;
			ret = epoll_ctl(epoll_fd, EPOLL_CTL_ADD, scratch_fd, &client_conn);
			if(ret < 0){
				perror("Epoll new client error\n");
				close(scratch_fd);
				break;
			}

			// Update conn state
			connections[next_open].active = 1;
			clock_gettime(CLOCK_REALTIME, &connections[next_open].last_access);
			connections[next_open].fd = scratch_fd;
			connections[next_open].data_length = 0;
			connections[next_open].buffer_size = 2048;

			// Info
			printf("Client %s connected in %d\n", inet_ntoa(connections[next_open].address.sin_addr), next_open);

			// Move next_open to first inactive connection (remains unchanged when full)
			for(long i = 1;i < MAX_CONNECTIONS;i++){
				if(connections[ (next_open + i) % MAX_CONNECTIONS ].active == 0){
					next_open = (next_open + i) % MAX_CONNECTIONS;
					break;
				}
			}
			num_conns++;
		}

		// Call epoll_wait2 with global timeout
		errno = 0;
		ret = epoll_pwait(epoll_fd, conn_events, MAX_EPOLL_EVENTS, epoll_timeout, NULL);
		if(ret < 0){
			printf("Thing");
			if(errno != EINTR) perror("Epoll wait error\n");
			break;
		}

		// Interact with any ready connections
		num_events = ret;
		for(int i = 0;i < num_events;i++){
			ret = handle_client(conn_events + i);
			if(ret > 0){
				printf("Server closing connection %s\n", inet_ntoa(connections[i].address.sin_addr));
				struct conn_state *client = (struct conn_state *) conn_events[i].data.ptr;
				// Remove connection from epoll
				ret = epoll_ctl(epoll_fd, EPOLL_CTL_DEL, client->fd, NULL);
				if(ret < 0){
					perror("Epoll server delete error\n");
					break;
				}
				client->active = 0;

				num_conns--;
				next_open = client - connections;
			}
		}

		// Check all connections for timeout (conn_timeout) and mark inactive. Close finished connections also?
		clock_gettime(CLOCK_REALTIME, &current_time);
		double res;
		for(long i = 0;i < MAX_CONNECTIONS;i++){
			// Only timeout active connections
			if(connections[i].active){
				res = timespecdiff(&current_time, &connections[i].last_access);
				if(res >= conn_timeout){
					printf("Client %s timed out\n", inet_ntoa(connections[i].address.sin_addr));

					// Remove from epoll first
					ret = epoll_ctl(epoll_fd, EPOLL_CTL_DEL, connections[i].fd, NULL);
					if(ret < 0){
						perror("Epoll timeout delete error\n");
						break;
					}

					// Close socket and clean up state
					close(connections[i].fd);
					connections[i].active = 0;
					next_open = i;
					num_conns--;
				}
			}
		}
	}

	/* Clean up
	*/
	sigaction(SIGINT, &old, NULL);

	// Close any client connections
	for(long i = 0;i < MAX_CONNECTIONS;i++){
		if(connections[i].active){
			close(connections[i].fd);
		}
	}

	// Close server socket
	printf("Closing socket..\n");
	close(server_sock);

	// Epoll end
	close(epoll_fd);

	return 0;
}
