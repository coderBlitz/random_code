#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<pthread.h>


// Maxes out CPU cores. More or less effective prank

void *loop(){//Function called by threads
	long i=0;
	while(1){
		i++;
	}
	printf("Thread done\n");
}

int main(){
	pthread_t pth1,pth2,pth3,pth4;
	pthread_create(&pth1,NULL,loop,NULL);//Starts second thread
	pthread_create(&pth2,NULL,loop,NULL);//Starts third thread
	pthread_create(&pth3,NULL,loop,NULL);//Starts fourth thread
	pthread_create(&pth4,NULL,loop,NULL);//Starts fifth thread

	printf("Main waiting for thread...\n");
//	pthread_join(pth,NULL);
	for(int i=1;i<=30;i++){
		printf("\rTime elapsed: %d",i);
		fflush(stdout);
		sleep(1);
	}
printf("\n");

	printf("Main and Thread done, Finishing...\n");
}
