/* By Chris Skane
Find perfect numbers up to a max value. Perfect numbers equal the sum of their factors
excluding the number itself. i.e 6=1+2+3, 28=1+2+4+7+14
The first 5 numbers, only 4 printed in current state, 6 28 496 8128 33,***,***
*/
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<signal.h>

time_t start;

void handler(int sig){
	printf("\nProgram interrupted, Time was %ld. Exiting...\n",time(0)-start);
	exit(0);
}

int main(){
	signal(SIGINT,handler);

	int *n=malloc(sizeof(int)); *n=2;
	int MAX=10000000;
	start=time(0);
	time_t avg=time(0);

	while((*n)<=MAX){
		int f=0;
		for(int j=1;j<(*n);j++){
			if((*n)%j==0){f+=j;}
		}
		if(f==(*n)){
			printf("\33[2K\r%d is a Perfect Number\n",*n);
		}
		if(((*n)%(MAX/200))==0){
printf("\r%ld:%ld %d%% through %d.  Average time per %d: %lds",(time(0)-start)/60,(time(0)-start)%60,(*n)/(MAX/100),MAX,(MAX/200),time(0)-avg);
			fflush(stdout);
			avg=time(0);
		}else{
			printf("\r%ldm%lds %d%% through %d.",(time(0)-start)/60,(time(0)-start)%60,(*n)/(MAX/100),MAX);
		}

		(*n)+=2;
	}
	free(n);
	printf("\nTotal time was %ld:%ld\n",(time(0)-start)/60,(time(0)-start)%60);
	exit(0);
}
