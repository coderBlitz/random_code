#include"device_launch_parameters.h"
#include "cuda_runtime.h"
#include<stdio.h>

int main(){
	cudaError_t err;
	err = cudaDeviceSynchronize();
	if(err != cudaSuccess){
		printf("%s\n",cudaGetErrorString(err));
		exit(1);
	}
	err = cudaDeviceReset();
	if(err != cudaSuccess){
		printf("%s\n",cudaGetErrorString(err));
		exit(1);
	}
}
