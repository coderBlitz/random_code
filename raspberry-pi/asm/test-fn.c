#include<stdio.h>
#include<stdlib.h>

extern long test(int,int);

int main(){
	printf("I said: ");
	fflush(stdout);
	long num = test(12,4);
	printf("!\n%ld\n", num);

	return 0;
}
