#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>

#include<sys/types.h>// For the networking
#include<sys/socket.h>// ""
#include <arpa/inet.h>

/* Format used for file transfer
   FILENAME - Max LENGTH characters
   FILELENGTH - N bytes
   BYTES PER TRANSFER - N bytes (try increments of 100 or something like that)
   FILE - This is where the actual file get sent
   [possibly something to show end of entire transfer]
*/

int main(){
	int LENGTH = 55;// Will be used to change filename length
	char FNAME[LENGTH+1];// Will store the filename sent by client
	unsigned int FILELENGTH;

	char *ip = "192.168.0.13";
	int port = 2000;// Port used for data transfer

	int server;
	server = socket(AF_INET,SOCK_STREAM,0);// socket(AF_INET,[stream type],[protocol, usually 0])

	struct sockaddr_in addr;
	addr.sin_family = AF_INET;// AF- Address Family. Just put this here
	addr.sin_addr.s_addr = inet_addr(ip);// For any interface/address
	addr.sin_port = htons(port);// The port on which to listen

	printf("File to Transmit: ");
	fgets(FNAME,LENGTH,stdin);
	FNAME[strlen(FNAME) - 1] = '\0';// Remove trailing newline

	FILE *fp = fopen(FNAME,"rb");
	if(fp == NULL){
		fprintf(stderr,"File \"%s\" does not exist!\n",FNAME);
		exit(1);
	}

	for(int i=LENGTH;i>=0;i--){
		if(FNAME[i] == '/') strcpy(FNAME,&FNAME[i+1]);
	}

	fseek(fp,0,SEEK_END);
	FILELENGTH = (unsigned int)ftell(fp);// Set the file length
	printf("File length: %u Bytes\n",FILELENGTH);
	rewind(fp);
//printf("FNAME: \"%s\"\n",FNAME);

	int err = connect(server,(struct sockaddr *)&addr,sizeof(struct sockaddr));// Accept connection
	if(err){
		fprintf(stderr,"Could not connect to server at \"%s\" on port %d. Stopping..\n",ip,port);
		close(server);
		exit(1);
	}
	printf("Successfully connected to server!\n");

//printf("[last]: %d\t[last-1]: %d\n",FNAME[strlen(FNAME)],FNAME[strlen(FNAME)-1]);
//				START FILE OPERATIONS HERE
printf("Sending \"%s\"\n",FNAME);

	short BPT;
	if(FILELENGTH <= 50) BPT = FILELENGTH;
	else BPT = 30;
	printf("BPT: %hd\n",BPT);

	write(server,FNAME,LENGTH);// Send filename
	write(server,&FILELENGTH,sizeof(unsigned int));// Send filelength
	write(server,&BPT,sizeof(short));// Send Bytes-per-Transfer

	long bytes_remain = FILELENGTH;
	char buffer[BPT];// Hold bytes until written to file
	while(1){
		if(bytes_remain <= 0) break;

		short bytes_read = fread(buffer,1,BPT,fp);// Get first part of file
		if(bytes_read != BPT) printf("Wrong number of bytes read. Probably EOF\n");

		short bytes_sent = write(server,buffer,bytes_read);// Send teh data
		if(bytes_sent != bytes_read) printf("Not all bytes read were sent!\n");

		bytes_remain -= bytes_read;
	}
	fclose(fp);// Close file
	printf("Transfer complete!\n");
//				    END FILE OPERATIONS

	close(server);// Cleanup
}
