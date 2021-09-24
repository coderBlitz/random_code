#include"gpio.h"
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<sys/time.h>

void ON(int g){
	GPIO_INP(g);
	GPIO_OUT(g);

	GPIO_SET = 1<<g;
}
void OFF(int g){
	GPIO_INP(g);
	GPIO_OUT(g);

	GPIO_CLR = 1<<g;
}

// G in GPIO number, cycle is integer percent, usec is microsecond
void duty(int g,unsigned int cycle, unsigned int usec){
	cycle = (cycle > 100)?100:cycle; // Can't have more than 100% duty cycle
	
	unsigned long rate = 10000; // 1/50hz = 0.02, or 20000 usec
	unsigned long cutoff = (unsigned long)(cycle*rate/100);
	unsigned long diff_usec = 0;
//	printf("Rate: %lu\tCutoff: %lu\n", rate, cutoff);

	struct timeval start, current, diff;
	gettimeofday(&start, NULL);
	gettimeofday(&current,NULL);
	timersub(&current,&start,&diff);

	while(diff_usec < usec){
		ON(g);
		usleep(cutoff);
		OFF(g);
		usleep(rate-cutoff);

		gettimeofday(&current,NULL);
		timersub(&current,&start,&diff);
		diff_usec = diff.tv_sec*1e6 + diff.tv_usec;
	}
}

int main(){
	setup_io();

	int pin = 6;

/*	ON(pin);
	sleep(1);
	OFF(pin);
*/
	int i =0;
/*	for(i = 1;i <= 100;i++){
		printf("Power %d%%\n", i);
		duty(pin,i,1e5);
	}*/


	return 0;
}
