#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"

#define N 22725
#define threads 1
// Max value is 16069 (64-bit/8-byte numbers)
// or 22725 (32-bit/4-byte numbers)

// GPU Max memory in bytes: 2065702500

__global__ void thing(int *arr){
	int UID = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;// 2-D Blocks, 1-D threads
//printf("UID: %d\n",UID);

	arr[UID] = UID;
}

int main(){
	uint X=N,Y=N;
	int *arr = (int *)malloc(threads*X*Y*sizeof(int));
	int *dev_arr;
	cudaError_t err;

	printf("Requesting %ld bytes of mems\n",threads*X*Y*sizeof(int));
	err = cudaMalloc(&dev_arr,threads*X*Y*sizeof(int));
	if(err != cudaSuccess){
		printf("ERROR: '%s'\n",cudaGetErrorString(err));
		exit(1);
	}

	size_t free_mem,total_mem;
	cudaMemGetInfo(&free_mem,&total_mem);
	printf("Total mem: %ldmb\nFree mem: %ldb\n",total_mem/(1024*1024),free_mem/(1));

	dim3 blocks(X,Y);// NxN grid
	thing<<<blocks,threads>>>(dev_arr);
	err = cudaDeviceSynchronize();
	if(err != cudaSuccess){
		printf("ERROR: '%s'\n",cudaGetErrorString(err));
	}

	err = cudaMemcpy(arr,dev_arr,threads*X*Y*sizeof(int),cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		printf("ERROR: '%s'\n",cudaGetErrorString(err));
	}

//	for(int i=0;i<X*Y*threads;i++) printf("arr[%ld]: %ld\n",i,arr[i]);

	free(arr);
	cudaDeviceReset();
}
