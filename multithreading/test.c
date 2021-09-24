#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<pthread.h>

/* Creates a 2-threaded program that prints 1-10
 The main prints odd numbers, the second thread prints even
*/


void *printStuff(){//Function called by second thread
	for(int i=2;i<=10;i+=2){
		printf("%d\n",i);
		sleep(1);
	}
	printf("Thread done\n");
}

int main(){
	pthread_t pth;
	//Starts second thread
	// ..._create(pthreadId,attribute,functionCalled,argumentPassedToFunction);
	pthread_create(&pth,NULL,printStuff,NULL);

	for(int i=1;i<=10;i+=2){
		printf("%d\n",i);
		sleep(1);
	}
	printf("Main waiting for thread...\n");
	pthread_join(pth,NULL);

	printf("Main and Thread done, Finishing...\n");
}
