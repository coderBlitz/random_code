#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"


__global__ void perfect(char *arr, long total, long offset){
	unsigned long id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;// 2-D Blocks, 1-D threads
	unsigned long N = id+offset, count=0;

	if(!(id > total)){ // Stops computation for threads outside of the range
		for(unsigned long i=1;i <= (unsigned long)sqrt((double)N); i++){
			if(N%i == 0) count++;
		}
//		printf("%d done!\n",id);
		arr[id] = ((count == 1)?1:0);// Sets to 1 if prime number, else 0
	}
	__syncthreads();
}

int main(){
	cudaDeviceReset();
	printf("Ready!\n");
	const int threads = 512;
	const int blocks = 1024;
	dim3 numBlocks(blocks,blocks);
	unsigned long SIZE = blocks*blocks*threads; // This is the total number of 
	unsigned long N = 1e9;
	printf("N: %lu\tblocks: %d\tthreads: %d\n",N,blocks,threads);

	char *primes = (char *)malloc(N*sizeof(char)); // Since we are only storing 1 or 0
	if(primes == NULL){
		printf("Too many primes, not enough mems.\n");
		exit(1);
	}
	for(unsigned long i=0;i<N;i++) primes[i] = 0;
	
//	printf("SET!\n");
	char *dev_array;
	
	cudaError_t err;
	err = cudaMalloc(&dev_array, N*sizeof(char));// Allocate memory for the array which will tell us which nums are prime
//	cudaDeviceSynchronize();
	if(err != cudaSuccess){
		printf("Damn it 1\n");
		printf("Error %s\n", cudaGetErrorString(err));
		cudaDeviceReset();
		free(primes);
		exit(1);
	}
//	printf("GOOOOO!\n");

	unsigned long count = 0, remain = SIZE;
	for (int i=0; count < N; i++){
		printf("Batch %d\n",i+1);
		if((count+remain) > N){// This makes sure that remain is never a negative number
			remain = N - count;
//			printf("REMAIN: %ld\tcount: %lu\n",remain,count);
		}

		perfect<<<numBlocks,threads>>>(dev_array, remain, count);// Run the GPU
		cudaDeviceSynchronize();
		if(err != cudaSuccess){
			printf("Error %s\n", cudaGetErrorString(err));
//			cudaDeviceReset();
//			free(primes);
//			exit(1);
		}

		err = cudaMemcpy(primes + count, dev_array, remain*sizeof(char), cudaMemcpyDeviceToHost); // Copy chunk into array
		if(err != cudaSuccess){
			printf("Damn it 2\n");
			printf("Error %s\n", cudaGetErrorString(err));
			cudaDeviceReset();
			free(primes);
			exit(1);
		}
		cudaDeviceSynchronize();

		count += remain;
	}
	
	printf("Counting..\n");
	count=0;
	for(unsigned long i=2; i<N; i++){// Checks if the number is prime, then adds to the total
		if(primes[i]){
//			printf("%lu is a prime number\n",i);
			count++;
		}
	}
	printf("There are %lu primes between 1 and %lu\n",count,N);
	
//	for(long i=0;i<10;i++) printf("primes[%d]: %d\n",837796+i,primes[837794+i]);

	free(primes);
	cudaDeviceReset();
}
