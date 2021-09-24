#include "cuda_runtime.h" //For the NVIDIA CUDA functionality
#include "device_launch_parameters.h"// ""

#include <stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<time.h>
#include<string.h>


//Simply has each CUDA block print off its number
__global__ void test(char **a,int len){
	int id = blockIdx.x;// blockDim is how many threads per block there are
	printf("%d (strlen %d): %s\n",id,len,a[id]);
}

void cpy(char *a,char *b){
	int i;
	for(i=0;a[i] != NULL;i++);
	i++;
	
	for(int n=0; b[n];i++,n++) a[i] = b[n];
	a[i] = '\0';// Null terminate
}

/*
char *fmt(char *a,int i){
	if(i < 0) return NULL;
	for(int n=0; i; n++){
		if(a[n]) i--;
	}
	
}*/
#define NUM 3
int main(){
	printf("And they're off!\n");
	
	
	char s[NUM*2 + NUM];
	
	
//	for(int i=0;i<3;i++) printf("Str[%d]: %s\n",i,s[i]);
	printf("sizeof(s): %ld\n",sizeof(s));
	
	strcpy(s,"hi");
	printf("s: %s\n",s);
	cpy(s,"by");
	printf("s[1]; %s\n",&s[3]);
	
//	int size = (sizeof(s))*sizeof(char);
//	char *dev_s;
	
//	cudaMalloc(&dev_s,size);
	
//	cudaMemcpy(dev_s, s, size,cudaMemcpyHostToDevice);
	
//	test <<<3,1>>>(dev_s,);
	
	cudaDeviceReset();
//	for(int i=0;i<3;i++) free(s[i]);
//	free(s);
	return 0;
}
