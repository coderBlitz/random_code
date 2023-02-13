/*
setsockopt (from ip(7))
	- IP_ADD_MEMBERSHIP : Join multicast group (ip_mreqn structure)
		- IP_ADD_SOURCE_MEMBERSHIP : Join, receive only from source
	- IP_DROP_MEMBERSHIP : Leave multicast group
	- IP_MULTICAST_LOOP
*/

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/ip.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char *argv[]){
	int sock = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);

	/// Prepare multicast address stuff
	struct ip_mreqn req;
	//req.imr_multiaddr.s_addr = ;
	const char *mcast_group = "239.123.213.1";
	inet_pton(AF_INET, mcast_group, &req.imr_multiaddr);
	req.imr_address.s_addr = INADDR_ANY; // Let system choose interface
	//inet_pton(AF_INET, "127.0.0.1", &req.imr_address);
	req.imr_ifindex = 0; // Any interface

	/// Join multicast group and disable packet looping
	int res = setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &req, sizeof(req));
	if(res == -1){
		perror("Setting opt failed");
		return -1;
	}
	char loop = 0;
	res = setsockopt(sock, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop));

	/// Bind to multicast port so packets can be received
	struct sockaddr_in outAddr;
	outAddr.sin_family = AF_INET;
	outAddr.sin_addr.s_addr = INADDR_ANY;
	outAddr.sin_port = htons(1234);
	res = bind(sock, (struct sockaddr*) &outAddr, sizeof(outAddr));
	if(res == -1){
		perror("Bind failed");
		return -1;
	}

	//sleep(5);

	const size_t buffer_len = 32;
	unsigned char buffer[buffer_len];

	/// Indefinitely listen on multicast socket
	outAddr.sin_addr.s_addr = req.imr_multiaddr.s_addr;
	while((res = read(sock, buffer, buffer_len-1)) >= 0){
		buffer[res] = 0;
		printf("Buffer: %s\n", buffer);

		res = sendto(sock, buffer, res, 0, (struct sockaddr*) &outAddr, sizeof(outAddr));
		if(res == -1){
			perror("Send failed");
		}
	}

	close(sock);

	return 0;
}
