#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>

#include<sys/types.h>// For the networking
#include<sys/socket.h>// ""
#include <arpa/inet.h>

/* Format used for file transfer
   FILENAME - Max 55 characters
   FILELENGTH - N bytes
   BYTES PER TRANSFER - N bytes (try increments of 100 or something like that)
   FILE - This is where the actual file get sent
   [possibly something to show end of entire transfer]
*/

int main(){
	char FNAME[56];// Will store the filename sent by client. Max length: 55
	long FILELENGTH;
	short BPT;
	FILE *fp;// Will write to local file as recieved from client

	int port = 2000;// Port used for data transfer

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

	err = listen(server,2);// Wait for connection
	if(err){
		fprintf(stderr,"Error listening\n");
		close(server);
		exit(1);
	}
	printf("Listening on port %d..\n",port);

	int client = accept(server,NULL,0);// Accept connection
	printf("Successfully connected to client!\n");

//				START FILE OPERATIONS HERE

	int recv = read(client,FNAME,55);// Get the filename (check for pre-existing files later)
//	File-existence check Yet-to-be implemented
	if(strlen(FNAME) <= 0){// If no name was recieved
		fprintf(stderr,"Did not recieve a filename. Stopping..\n");
		close(client);
		close(server);
		exit(1);
	}
	fp = fopen(FNAME,"wb");// Open file to write
	printf("Starting transfer of file \"%s\"...\n",FNAME);

	recv = read(client,&FILELENGTH,sizeof(long));// U_INT can handle file up to ~4.2GB (wouldn't try)
	if(recv != sizeof(long) || FILELENGTH <= 0){// If something happened with transfer
		fprintf(stderr,"Did not recieve proper filelength. Stopping..\n");
		close(client);
		close(server);
		exit(1);
	}
	printf("File Size: %u Bytes\n",FILELENGTH);

	recv = read(client,&BPT,sizeof(short));// Short because network packets can only be so big, why waste space

	if(recv != sizeof(short) || BPT <= 0 || BPT >= 200){
		fprintf(stderr,"Did not recieve proper Bytes-per-transfer number (%hd). Stopping..\n",BPT);
		close(client);
		close(server);
		exit(1);
	}
	printf("%hd Bytes-per-Transfer\n",BPT);

	long bytes_remain = FILELENGTH;
	char buffer[BPT];// Hold bytes until written to file
	while(1){
		if(bytes_remain <= 0) break;
//		printf("\rbytes_remain: %ld",bytes_remain);

		int bytes_read = read(client,buffer,BPT);// Read into buffer
//		if(bytes_read != (int)BPT) fprintf(stderr,"Incorrect number of bytes recieved(%d) or EOF\n",bytes_read);

		fwrite(buffer,1,bytes_read,fp);// Write to file

		bytes_remain -= bytes_read;// Just in case bytes_read doesn't match BPT
	}
	fclose(fp);// Close file
	printf("\nTransfer complete!\n");
//				    END FILE OPERATIONS

	close(client);
	close(server);// Cleanup
}
