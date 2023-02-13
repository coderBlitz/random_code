#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char *argv[]){
	int sock = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
	int res;

	const size_t msg_len = 64;
	char msg[msg_len];
	snprintf(msg, msg_len, "This is my message!");

	struct sockaddr_in peer;
	peer.sin_family = AF_INET;
	peer.sin_port = htons(12323);
	res = inet_pton(AF_INET, "127.0.0.1", &peer.sin_addr);
	if(res == -1){
		perror("Address conversion failed");
		return -1;
	}
	//peer.sin_addr.s_addr = htonl;

	res = bind(sock, (struct sockaddr*) &peer, sizeof(peer)); // Shows listening in ss
	if(res == -1){
		perror("Bind failed");
		return -1;
	}
	printf("Socket bound. Check ss.\n");

	sleep(5);

	res = connect(sock, (struct sockaddr*) &peer, sizeof(peer)); // Shows connection in ss
	if(res == -1){
		perror("Failed to connect socket");
		close(sock);
		return -1;
	}
	printf("Socket connected.\n");

	ssize_t count;
	for(int i = 0;i < 10;i++){
		count = sendto(sock, msg, msg_len, 0, (struct sockaddr*) &peer, sizeof(peer));
		if(count == -1){
			perror("Send failed");
		}
	}

	sleep(3);

	peer.sin_family = AF_UNSPEC;
	res = connect(sock, (struct sockaddr*) &peer, sizeof(peer)); // Removes from ss connections
	if(res == -1){
		perror("Failed to connect socket");
		close(sock);
		return -1;
	}
	printf("Socket dissociated??\n");

	peer.sin_family = AF_INET; // Verifying socket still usable
	count = sendto(sock, msg, msg_len, 0, (struct sockaddr*) &peer, sizeof(peer));
	if(count == -1){
		perror("Send failed");
	}

	sleep(3);

	close(sock);
	printf("Socket closed. Check ss and stuff.\n");

	sleep(5);

	return 0;
}
