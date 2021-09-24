#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<pthread.h>

static pthread_barrier_t bar;// The Barrier. static keeps scope to this file only

void *printStuff(){//Function called by second thread
	for(int i=1;i<=10;i++){
		printf("%d\n",i);
		sleep(1);
	}

	pthread_barrier_wait(&bar);// Wait on barrier. Mostly meant to synchronize with main()
	sleep(1);

	for(int i=11;i<=15;i++){
		printf("%d\n",i);
		sleep(1);
	}
	printf("Thread done\n");
}

int main(){
	pthread_t pth;

	pthread_barrier_init(&bar,NULL,2);// Barrier for main to wait for thread to get to

	pthread_create(&pth,NULL,printStuff,NULL);// start thread

	printf("Waiting for thread to print numbers 1-10...\n");
	pthread_barrier_wait(&bar);// Wait for barrier

	printf("Done. Waiting to print 11-15 and finish...\n");
	pthread_join(pth,NULL);

	printf("Thread done. Main exiting..\n");
}
