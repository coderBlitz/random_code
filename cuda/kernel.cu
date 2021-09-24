#include<stdio.h>
#include<stdlib.h>
#include "cuda_runtime.h" //For the NVIDIA CUDA functionality
#include "device_launch_parameters.h"// ""


/* File name can be *.cu
   Use developer command prompt, and compile with "nvcc filename.cu"
*/

__global__ void multiplyMtx(int* a,int* b,int* c,int* n)
{
	int id = blockIdx.x;
	if (id < *n){
		c[id] = a[id] * b[id];
	}
}

int main()
{
	const int N = 4;
	int a[N] = {1,4,2,9};
	int b[N] = {8,3,0,6};
	int sum[N];
	
	int* dev_a;
	int* dev_b;
	int* dev_c;
	int* max;

	cudaMalloc(&dev_a, N * sizeof(int));
	cudaMalloc(&dev_b, N * sizeof(int));
	cudaMalloc(&dev_c, N * sizeof(int));
	cudaMalloc(&max, N * sizeof(int));

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(max, &N, N * sizeof(int), cudaMemcpyHostToDevice);

	multiplyMtx <<<N,1>>>(dev_a,dev_b,dev_c,max);

	cudaMemcpy(sum, dev_c, N * sizeof(int) ,cudaMemcpyDeviceToHost);

	for (int j = 0; j < N; j++){
		printf("Value at product[%d]: %d\n", j, sum[j]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(max);
	cudaDeviceReset();
    return 0;
}
