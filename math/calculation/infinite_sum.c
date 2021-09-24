#include<stdio.h>

int main(){
	double sum = 0;
	unsigned long long count = 1;
	while(1){
		sum += 1.0/count;
		count *= 2;
		printf("\r%lf",sum);
	}
}
