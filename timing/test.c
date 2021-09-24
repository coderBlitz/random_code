#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<sys/time.h>

int main(){
	struct timeval tval_start,tval_end,tval_diff;

	gettimeofday(&tval_start,NULL);// Get start execution time
	sleep(2);// Wait 2 seconds
	gettimeofday(&tval_end,NULL);

	timersub(&tval_end,&tval_start,&tval_diff);// Get difference

	printf("Time elapsed: %ld.%06ld\n",(long int)tval_diff.tv_sec,tval_diff.tv_usec);
}
