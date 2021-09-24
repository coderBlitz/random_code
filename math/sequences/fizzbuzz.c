#include<stdio.h>

int main(){
	int N = 60;

	int ctr3 = 3;
	int ctr5 = 5;

	for(int i = 1;i <= N;++i){
		if((ctr3 == ctr5) && (i == ctr3)){
			printf("Fizz-Buzz\n");
			ctr3 += 3;
			ctr5 += 5;
		}else if(i == ctr3){
			printf("Fizz\n");
			ctr3 += 3;
		}else if(i == ctr5){
			printf("Buzz\n");
			ctr5 += 5;
		}else{
			printf("%u\n", i);
		}
	}

	return 0;
}
