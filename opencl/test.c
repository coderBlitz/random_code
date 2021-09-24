#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<CL/cl.h>

#define MAX_SOURCE (0x100000)

int main(int argc, char *argv[]){
	unsigned int N = 256;

	int *A = malloc(N * sizeof(*A));
	if(A == NULL){
		fprintf(stderr, "Failed to malloc A\n");
		exit(1);
	}
	int *B = malloc(N * sizeof(*A));
	if(A == NULL){
		fprintf(stderr, "Failed to malloc A\n");
		exit(1);
	}

	int *C = malloc(N * sizeof(*A));
	if(A == NULL){
		fprintf(stderr, "Failed to malloc A\n");
		exit(1);
	}

	for(int i = 0;i < N;++i){
		A[i] = i;
		B[i] = -i;
	}

	FILE *fp = fopen("vector_add_kernel.cl", "r");
	if(fp == NULL){
		fprintf(stderr, "Failed to open kernel file\n");
		exit(2);
	}
	char *source = malloc(MAX_SOURCE);
	if(source == NULL){
		fprintf(stderr, "Could not mallocate source\n");
		exit(1);
	}
	size_t source_size = fread(source, 1, MAX_SOURCE, fp);
	fclose(fp);

	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint num_devs;
	cl_uint num_plats;
	cl_int ret;

	clGetPlatformIDs(1, &platform_id, &num_plats);
	clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devs);

	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &ret);

	cl_mem A_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(*A), NULL, &ret);
	cl_mem B_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(*B), NULL, &ret);
	cl_mem C_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(*C), NULL, &ret);

	clEnqueueWriteBuffer(queue, A_mem, CL_TRUE, 0, N*sizeof(*A), A, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, B_mem, CL_TRUE, 0, N*sizeof(*B), B, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, &source_size, &ret);
	clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

	clSetKernelArg(kernel, 0, sizeof(A_mem), &A_mem);
	clSetKernelArg(kernel, 1, sizeof(B_mem), &B_mem);
	clSetKernelArg(kernel, 2, sizeof(C_mem), &C_mem);

	size_t global_size = N;
	size_t item_size = 64;
	ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &item_size, 0, NULL, NULL);
	if(ret){
		printf("Kernel failed\n");
	}

	clEnqueueReadBuffer(queue, C_mem, CL_TRUE, 0, N*sizeof(*C), C, 0, NULL, NULL);

	for(int i = 0;i < N;++i){
		printf("%d * %d = %d\n", A[i], B[i], C[i]);
	}

	clFlush(queue);
	clFinish(queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(C_mem);
	clReleaseMemObject(B_mem);
	clReleaseMemObject(A_mem);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	free(A);
	free(B);
	free(C);
	return 0;
}
