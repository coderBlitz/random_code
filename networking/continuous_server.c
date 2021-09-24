#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<fcntl.h>
#include<time.h>

#include<sys/types.h>// For the networking
#include<sys/socket.h>// ""
#include <arpa/inet.h>

/* Program will be able to listen for incoming connections without using
   more than a single thread. Connect using "./message_client"
*/


int main(){
	char buffer[151];// Will store the message sent by client
	int port = 2000;

	int server;
	server = socket(AF_INET,SOCK_STREAM,0);// socket(AF_INET,[stream type],[protocol, usually 0])

	int flags;
	if(-1 == (flags = fcntl(server,F_GETFL,0))) flags=0;
	fcntl(server,F_SETFL, flags | O_NONBLOCK);// Set non-blocking to allow loop to continue w/o client

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

	err = listen(server,1);// Wait for connection
	if(err){
		fprintf(stderr,"Error listening\n");
		close(server);
		exit(1);
	}

	printf("Listening on port %d..\n",port);

	time_t temp=time(0);
	int client=-1;
	int delay=7;
	while(1){
		if(time(0) - temp >= delay && client == -1){
			client = accept(server,NULL,NULL);// Accept connection if possible, otherwise move on
			if(client == -1){
		//		printf("No client available... Re-trying in %d seconds\n",delay);
			}
			else{
				if(-1 == (flags = fcntl(server,F_GETFL,0))) flags=0;
				fcntl(client,F_SETFL, flags | O_NONBLOCK);// Set non-blocking to let loop run
			}
			fflush(stdout);
			temp = time(0);
		}

		while((err = read(client,buffer,150)) > 0) printf("%s",buffer);
//		printf("Closing client \"%d\"\n",client);
		close(client);
		client = -1;

		sleep(1);
	}


	close(server);// Cleanup
}

