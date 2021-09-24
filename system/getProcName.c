#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main(){
	char *procFile = malloc(19 * sizeof(char));// Room for 5-digit PID
	strcat(procFile,"/proc/");

	char pid[5];
	int tmpPID;

	printf("Enter PID to get process name: ");
	scanf("%d",&tmpPID);// Take input as int to prevent non-numbers

	sprintf(pid,"%d",tmpPID);// Put PID into string

	strcat(procFile,pid);// Add pid to file path
	strcat(procFile,"/status");// go to status to retrieve name
//	printf("Proc file is %s\n",procFile);

	FILE *fp = fopen(procFile,"r");
	if(fp == NULL){// Check if file exists or not
		printf("There is no process with the PID %s\n",pid);
		exit(1);
	}

	char c;
	for(int i=0;i<6;i++) c = getc(fp);// Skip to proccess name

	while((c = getc(fp)) != '\n') printf("%c",c);// Print name
	printf("\n");// Print newline after name
}
