/* Attempt to bind a privileged port.
*/

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char *argv[]){
	int sock = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
	if(sock == -1){
		perror("Socket creation failed");
		return -1;
	}

	int res;

	// Setup addr info
	struct sockaddr_in peer;
	socklen_t peer_len = sizeof(peer);
	peer.sin_family = AF_INET;
	peer.sin_port = htons(597);
	res = inet_pton(AF_INET, "127.0.0.1", &peer.sin_addr);
	//peer.sin_addr.s_addr = INADDR_LOOPBACK;

	// Bind
	res = bind(sock, (struct sockaddr*) &peer, sizeof(peer));
	if(res == -1){
		perror("bind failed");
		close(sock);
		return -1;
	}

	// Cleanup stuff
	close(sock);

	return 0;
}
