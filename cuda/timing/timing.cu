#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<unistd.h>

__global__ void thing(double nums,double *res){
	*res = nums*nums;
}

int main(){
	struct timeval start,end,diff;
	int N = 100000;
	double n[N],res=0.0;
	for(int i=0;i<N;i++) n[i] = i+1;

	double *d_res;
	cudaMalloc(&d_res,sizeof(res));

	gettimeofday(&start,NULL);

	for(int i=0;i<N;i++){
		thing<<<1,1>>>(n[i],d_res);
		cudaMemcpy(&res,d_res,sizeof(res),cudaMemcpyDeviceToHost);
		printf("Ans (%d): %lf\n",i+1,res);
		fflush(stdout);
	}
	gettimeofday(&end,NULL);

	timersub(&end,&start,&diff);

	printf("\n%ld.%06lds\n",res,diff.tv_sec,diff.tv_usec);

	cudaDeviceReset();
}
