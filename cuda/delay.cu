#include"cuda_runtime.h"
#include"device_launch_parameters.h"

#include<stdio.h>
#include<unistd.h>
#include<sys/time.h>

__global__ void kernel(){
	printf("Hi\n");
}

int main(){
	printf("Starting\n");


	struct timeval start,end,result;

	//Preliminary call just in case
	gettimeofday(&start,NULL);
	kernel<<<4,1>>>();
	gettimeofday(&end,NULL);

	int err = cudaDeviceSynchronize();
	if(err != cudaSuccess) printf("Something failed\n");

	timersub(&end,&start,&result);
	printf("Run 1: %ld.%.4ld\n",result.tv_sec,result.tv_usec);


	printf("Wait two..\n");
	sleep(2);

	gettimeofday(&start,NULL);
	kernel<<<4,1>>>();
	gettimeofday(&end,NULL);
	timersub(&end,&start,&result);
	printf("Run 2: %ld.%.4ld\n",result.tv_sec,result.tv_usec);


	cudaDeviceReset();
	printf("Done!\n");
}
