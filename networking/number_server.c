#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

#include<sys/types.h>// For the networking
#include<sys/socket.h>// ""
#include <arpa/inet.h>

int main(){
	int buffer;// Will store the number sent by client
	int port = 2000;

	int server;
	server = socket(AF_INET,SOCK_STREAM,0);// socket(AF_INET,[stream type],[protocol, usually 0])

	struct sockaddr_in addr;
	addr.sin_family = AF_INET;// AF- Address Family. Just put this here
	addr.sin_addr.s_addr = htonl(INADDR_ANY);// For any interface/address
	addr.sin_port = htons(port);// The port on which to listen

	int err = bind(server,(struct sockaddr *)&addr,sizeof(struct sockaddr));// Bind to port
	if(err){// Catch the error
		fprintf(stderr,"Unable to bind on port %d\n",port);
		close(server);
		exit(1);
	}

	printf("Listening on port %d..\n",port);
	err = listen(server,1);// Wait for connection
	if(err){
		fprintf(stderr,"Error listening\n");
		close(server);
		exit(1);
	}

	int client = accept(server,NULL,0);// Accept connection

	int total = read(client,&buffer,sizeof(buffer));// Get number
	close(client);// Close the connection after message recieved
	printf("Recieved %d bytes\n",total);// Should be 4 bytes for int


	printf("Number sent: %d\n",buffer);// Print Number

	close(server);// Cleanup
}
