#include<stdio.h>
#include<stdlib.h>

int main(){
	unsigned long long high = 10000000;
	int *lens = malloc(high*sizeof(int));
	unsigned long long max=0;

	for(unsigned long long i=0;i<high;i++){
		int count=0;
		unsigned long long num=i;
		while(num > 1){
			if(num%2 == 0) num /= 2;
			else num = num*3 + 1;
			count++;
		}
		lens[i] = count;
		if(lens[i] > lens[max]) max = i;
	}
	printf("Largest chain: %d (%llu)\n",lens[max],max);


	free(lens);
}
