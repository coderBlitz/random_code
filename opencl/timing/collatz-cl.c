#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<time.h>
#include<CL/cl.h>

#define MAX_SOURCE (0x100000)

void timespecsub(struct timespec *a, struct timespec *b, struct timespec *res){
	res->tv_sec = a->tv_sec - b->tv_sec;
	res->tv_nsec = a->tv_nsec - b->tv_nsec;
	if(res->tv_nsec < 0){
		res->tv_nsec += 1000000000;
		res->tv_sec -= 1;
	}
}

int main(int argc, char *argv[]){
	unsigned long N_max = 0x40 << 22;
	struct timespec start, end, diff;

	int *results = malloc(N_max * sizeof(*results));
	if(results == NULL){
		fprintf(stderr, "Failed to malloc A\n");
		exit(1);
	}
	for(int i = 0;i < N_max;++i) results[i] = 0;

	FILE *fp = fopen("collatz.cl", "r");
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
	if(device_id == NULL){
		fprintf(stderr, "No device found\n");
		free(results);
		exit(2);
	}

	size_t maxGroup = 0;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxGroup), &maxGroup, NULL);
	printf("Max group size: %d\n", maxGroup);

	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &ret);

	cl_mem results_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N_max * sizeof(*results), NULL, &ret);

	//clEnqueueWriteBuffer(queue, A_mem, CL_TRUE, 0, N*sizeof(*A), A, 0, NULL, NULL);

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, &source_size, &ret);
	clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	cl_kernel kernel = clCreateKernel(program, "collatz", &ret);

	clSetKernelArg(kernel, 0, sizeof(results_mem), &results_mem);

	for(unsigned long N = 0x40;N <= N_max;N <<= 1){
		//printf("Max number: %u\n", N);
		printf("%u,", N);

		clock_gettime(CLOCK_REALTIME, &start);
		size_t global_size = N;
		size_t item_size = 64;
		ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &item_size, 0, NULL, NULL);
		if(ret){
			printf("Kernel failed\n");
		}

		clEnqueueReadBuffer(queue, results_mem, CL_TRUE, 0, N*sizeof(*results), results, 0, NULL, NULL);

		unsigned int max = 0;
		unsigned int maxPos = -1;
		for(int i = 0;i < N;++i){
			//printf("%d = %d\n", i, results[i]);
			if(results[i] > max){
				max = results[i];
				maxPos = i;
			}
		}
		clock_gettime(CLOCK_REALTIME, &end);
		timespecsub(&end, &start, &diff);

		//printf("Max chain length is %u, from %u in %ld.%09lds\n", max, maxPos, diff.tv_sec, diff.tv_nsec);
		printf(" %u, %u, %ld.%09ld\n", max, maxPos, diff.tv_sec, diff.tv_nsec);
	}


	clFlush(queue);
	clFinish(queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(results_mem);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	free(results);
	return 0;
}
