/*	Conditional call comparison
	Test a large conditional block for two sections of code, against a small
	 conditional that modifies a function pointer.
*/

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

int fn1(long x){

}

int main(int argc, char *argv[]){


	return 0;
}
