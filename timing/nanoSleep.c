#include<stdio.h>
#include<stdlib.h>
#include<time.h>

void timespecsub(struct timespec *a, struct timespec *b, struct timespec *res){
	res->tv_sec = a->tv_sec - b->tv_sec;
	res->tv_nsec = a->tv_nsec - b->tv_nsec;
	if(res->tv_nsec < 0){
		res->tv_nsec += 1000000000;
		res->tv_sec -= 1;
	}
}

int main(){
	struct timespec t;
	t.tv_sec = 0;
	t.tv_nsec = 500000000;

	nanosleep(&t,NULL);
	printf("Done\n");
}
