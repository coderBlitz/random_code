#include<stdio.h>
#include<stdlib.h>

int main(int argc,char **argv){
	if(argc == 1 || argc == 2){ printf("summation START_NUM END_NUM\n"); exit(0); }

	char *p;
	int start = (int)strtol(argv[1],&p,10);
	int end = (int)strtol(argv[2],&p,10);

	int n = (end-start)+1;

	long sum = (n/2.0)*(start+end);

	printf("%ld\n",sum);
}
