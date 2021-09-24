#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>

#include<sys/types.h>// For the networking
#include<sys/socket.h>// ""
#include <arpa/inet.h>// ""

int main(){
	int msg = 24;// Message being sent
	char *ip = "127.0.0.1";
	int port = 2000;

//	printf("Enter message to be sent: ");
//	fgets(msg,100,stdin);

	int client;
	client = socket(AF_INET,SOCK_STREAM,0);// socket(AF_INET,[stream type],[protocol, usually 0])

	struct sockaddr_in addr;
	addr.sin_family = AF_INET;// AF- Address Family. Just put this here
	addr.sin_addr.s_addr = inet_addr(ip);// The address to connect to
	addr.sin_port = htons(port);// The port on which to connect

	printf("Attempting to connect to \"%s\" on port %d...\n",ip,port);
	int err = connect(client,(struct sockaddr *)&addr,sizeof(struct sockaddr));// Connect to server
	if(err){// Get the exit status to determine successful connection
		fprintf(stderr,"Could not connect to \"%s\" on port %d\n",ip,port);
		close(client);
		exit(1);
	}
	printf("Connected!\n");

	printf("Sending message..\n");
	int sent = write(client,&msg,sizeof(msg));// Send message
	if(sent != sizeof(msg)) fprintf(stderr,"Something happened while sending the number!\n");// Error
	else printf("Number sent!\n");// Normal

	close(client);// Close stream
}
