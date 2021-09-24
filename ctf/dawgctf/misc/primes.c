#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main(){
	unsigned long long max = 11000000;
	unsigned long count=0;
	printf("GO!\n");
	for(unsigned long long i=10000000;i<max;i++){
		int sum = 0;
		for(unsigned long n=1;n<(unsigned long)sqrt(i)+1;n++){
			if(i%n == 0) sum += n;
			if(sum > 1) break;// If it is greater than 1, it is not prime so no need to go on
		}
		if (sum == 1) count++;
		else continue;
	}
	printf("There are %d prime numbers up to %d\n",count,max);
}
