#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char *argv[]){
	int sock = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0); // Shows up with `ss -au` as unconnected socket

	printf("Socket open. Check ss, netstat, or whatever. Sleeping for 5 seconds.\n");

	const size_t msg_len = 64;
	char msg[msg_len];
	snprintf(msg, msg_len, "This is my message!");

	struct sockaddr_in peer;
	peer.sin_family = AF_INET;
	peer.sin_port = htons(12323);
	//int res = inet_pton(AF_INET, "xxx.xxx.xxx.xxx", &peer.sin_addr);
	peer.sin_addr.s_addr = INADDR_LOOPBACK;

	ssize_t count;
	for(int i = 0;i < 10;i++){
		count = sendto(sock, msg, msg_len, 0, (struct sockaddr*) &peer, sizeof(peer));
		if(count == -1){
			perror("Send failed");
		}
	}

	sleep(10);

	close(sock);

	return 0;
}
