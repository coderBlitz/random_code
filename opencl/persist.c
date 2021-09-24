#include<stdio.h>
#include<stdlib.h>
#include<CL/cl.h>

#define MAX_SOURCE (0x1000)

int main(int argc, char *argv[]){
	unsigned int N = 64;

	int *A = malloc(N * sizeof(*A));
	int *B = malloc(N * sizeof(*A));

	for(unsigned int i = 0;i < N;++i){
		A[i] = i;
		B[i] = 0;
	}

	char *source = malloc(MAX_SOURCE * sizeof(*source));
	size_t source_size = 0;

	FILE *fp = fopen("persist_kernel.cl", "r");
	if(fp == NULL){
		fprintf(stderr, "Could not open file\n");
		exit(1);
	}
	source_size = fread(source, sizeof(*source), MAX_SOURCE, fp);
	fclose(fp);

	cl_platform_id platform;
	cl_device_id device;
	cl_uint num_plats;
	cl_uint num_devs;
	cl_int ret;

	clGetPlatformIDs(1, &platform, &num_plats);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &num_devs);

	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, &ret);

	cl_mem A_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(*A), NULL, &ret);
	cl_mem B_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N*sizeof(*B), NULL, &ret);

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, (const size_t *)&source_size, &ret);
	clBuildProgram(program, 1, &device, NULL, NULL, NULL);

	cl_kernel kernel = clCreateKernel(program, "persist", &ret);

	clEnqueueWriteBuffer(queue, A_mem, CL_TRUE, 0, N * sizeof(*A), A, 0, NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(A_mem), &A_mem);
	clSetKernelArg(kernel, 1, sizeof(B_mem), &B_mem);

	size_t total = N;
	size_t wg = 64;
	ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &total, &wg, 0, NULL, NULL);
	if(ret){
		fprintf(stderr, "Kernel failed");
	}
	ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &total, &wg, 0, NULL, NULL);
	if(ret){
		fprintf(stderr, "Kernel failed");
	}

	clEnqueueReadBuffer(queue, B_mem, CL_TRUE, 0, N * sizeof(*B), B, 0, NULL, NULL);

	for(int i = 0;i < N;++i){
		printf("B[%d]: %d\n", i, B[i]);
	}


	clFlush(queue);
	clFinish(queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(B_mem);
	clReleaseMemObject(A_mem);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseDevice(device);
	free(source);
	free(B);
	free(A);
	return 0;
}
