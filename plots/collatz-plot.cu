#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
//#include<math.h>
//#include<time.h>

/* For starting number N, if N is odd then 3N+1, if N is even N/2
   Program finds chain length for each number, and returns highest
   *Note* The '+2' in multiple places is because I'm skipping 0 and 1 (not anymore)
*/

#define NUM_BLOCKS 5669 // 22725 max using 1 thread. 16069 using 2 (205710954 Bytes of memory to work with)
//#define NUM_BLOCKS 4
#define threads 16 // Try and keep threads powers of 16 (powers of 2 at minimum)

__global__ void chains(int *arr, long TOTAL, unsigned long OFFSET){
	unsigned int UID = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;// 2-D Blocks, 1-D threads
	unsigned long N = UID + OFFSET;// N is the number computed, UIDgives position OFFSET is for multiple runs
	unsigned long count=0; // Running total for compute loop

//	printf("UID: %u\nN: %lu\nOFFSET: %lu\nMAX: %lu\n",UID, N, OFFSET, MAX);
//	if(UID == TOTAL+1) printf("UID: %u\nN: %lu\nOFFSET: %lu\nMAX: %lu\n",UID, N, OFFSET, MAX);

// The only reason I check for this is because the block dimensions dont change with N
// Hence why we only need to check the UID for being to large, because the array size remains the same.
	if (!(UID > TOTAL)){
		while(N > 1){// Calculate
			if (N == 0){
				count = 0;
				break;
			}
			if(N%2 == 0) N /= 2;
			else N = 3*N + 1;
			count++;
		}
		arr[UID] = count;// Store result in array
	}
	__syncthreads();
}

int main(){
	cudaDeviceReset();
	cudaError_t err;
	printf("Ready!\n");

	size_t free_mem,total;
	cudaMemGetInfo(&free_mem,&total);// Get memory stats
//	printf("\nTotal mem: %ld\nFree mem: %ld\n",total,free_mem);// For debug purposes

	dim3 blocks(NUM_BLOCKS,NUM_BLOCKS);// Define an NxN 2D grid
	long TOTAL = NUM_BLOCKS*NUM_BLOCKS*threads;// Simplify total size for each chunk (mostly for GPU)
	unsigned long N = 1800000000;// The number of numbers we compute to (1.84bil uses ~7G of memory)
	unsigned long count = 0;

	int *len_array = (int *)malloc(N * sizeof(int));// Mem for host array
	if (len_array == NULL){
		printf("N = %lu is too large, not enough memory. Lowering N..\n",N);
		while (len_array == NULL){
			len_array = (int *)malloc(N*sizeof(int));
			N -= N/100;
		}
//		exit(1);
	}

	for(unsigned long i=0;i<N;i++) len_array[i] = 0;
	
	int *dev_array;// Ptr for GPU


	cudaMemGetInfo(&free_mem,&total);// Get memory stats
	err = cudaMalloc(&dev_array,TOTAL*sizeof(int));// Allocate GPU mem for array
	if(err != cudaSuccess){// Error checking
		printf("Damn it: '%s'\n",cudaGetErrorString(err));
		cudaFree(dev_array);

		TOTAL -= (TOTAL*4 - free_mem);// Should give us max memory usage per chunk
		printf("Lowering mems...\nTotal now: %ld\nSize: %ld\n",TOTAL,TOTAL*4);
		err = cudaMalloc(&dev_array,TOTAL*sizeof(int));
		if(err != cudaSuccess){// Error checking
			printf("Damn it again: '%s'\n",cudaGetErrorString(err));
			exit(1);
		}
//		cudaFree(dev_array);
	}

	printf("Range 0-%lu\n", N);
	unsigned long temp = 0, REMAIN = TOTAL;
	for(int i=0;count < N;i++){
		printf("Run %d\n",i+1);

		temp = count;// Hold previous value
		if ((count+TOTAL) > N){
			printf("Finishing last run..\n");
			REMAIN = N - count;// Makes sure last copy doesn't try to copy more than host array size
			count = N;// Add last little bit to count
		}

//	printf("Running..\nCOUNT=%lu\n",count);
	printf("Running..\n");

		err = cudaDeviceSynchronize();
		if(err != cudaSuccess){
			printf("Damn it, kernel broke: '%s'\n",cudaGetErrorString(err));
		}

		chains<<<blocks,threads>>>(dev_array, REMAIN, temp);// GPU call (number array, chunk size, offset count)

		err = cudaDeviceSynchronize();// Wait for everything to finish
//		cudaGetLastError();
		if(err != cudaSuccess){
			printf("Damn it, kernel broke: '%s'\n",cudaGetErrorString(err));
		}

		count += REMAIN;// We do TOTAL numbers of calculations per run up until N

		err = cudaMemcpy(len_array + temp, dev_array, REMAIN*sizeof(int), cudaMemcpyDeviceToHost);// Get data back
		if(err != cudaSuccess){
			printf("Damn it: '%s'\n",cudaGetErrorString(err));
			exit(1);
		}

		if (temp > count){ // Should catch overflow
			printf("Overflow, so we're done here\n");
			break;
		}
		
		err = cudaDeviceSynchronize();
		if(err != cudaSuccess){
			printf("Damn it, kernel broke: '%s'\n",cudaGetErrorString(err));
		}
	}

	printf("Counting then plotting..\n");

	FILE *plot = popen("gnuplot -p","w");
	fprintf(plot, "set title 'C collatz chain length'\n");
	fprintf(plot, "set terminal qt size 1280,720\n");
	fprintf(plot, "plot '-'\n");

	unsigned long max=0;
	for(unsigned long i=0;i<N;i++){
		if(len_array[i] > len_array[max]) max = i;// Find max
		fprintf(plot, "%lu %lu\n", i, len_array[i]);
	}
	fprintf(plot, "e");
	fclose(plot);// Close gnuplot

//for(int i=268549000;i<268550000;i++) printf("%d- %lu\n",i,len_array[i]);// Debug print
	printf("Longest chain: %lu with a length of %d\n",max,len_array[max]);// Result


	free(len_array);// Clean up
	cudaFree(dev_array);
//	cudaFree(dev_max);
	cudaDeviceReset();// Make sure we don't have memory leaks or anything like that
}
