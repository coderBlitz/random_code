#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>

static int nums[1000];
static int products[1000];
static FILE *fp;

void *half(){
	for(int i=0;i<500;i++){
		products[i] = nums[i] * nums[i+1];
	}
	printf("0-499 Done!\n");

	for(int i=0;i<500;i++) fprintf(fp,"%d\n",products[i]);// Write first half
}

int main(){
	srand(time(0));
	for(int n=0;n<1000;n++) nums[n] = rand() % 50;// Fill with rand nums

	pthread_t pth;
	fp = fopen("/tmp/nums.txt","w");

	pthread_create(&pth,NULL,&half,NULL);

	for(int i=500;i<1000;i++) products[i] = nums[i] * nums[i-1];
	printf("500-999 Done!\n");

	pthread_join(pth,NULL);
	for(int i=500;i<1000;i++) fprintf(fp,"%d\n",products[i]);// Write rest


	fclose(fp);
}
