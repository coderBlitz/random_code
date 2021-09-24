#include<stdio.h>
#include<math.h>

// Find Nth number in Fibonacci sequence

int main(){
	double Phi = (1 + sqrt(5))/2;
	double phi = 1/Phi;

	unsigned int num=0;
	printf("Fibonacci number finder\nEnter number: ");
	scanf("%u",&num);

	if(num > 93) printf("Number to large! Try again\n");
	
	unsigned long long result = 0,tmp=0,tmp2=1;
//	result = (pow(Phi,(double)num) - pow((-phi),(double)num))/sqrt(5);// Approx.
	for(int i=0;i<num-1;i++){
		result = tmp + tmp2;
		tmp = tmp2;
		tmp2 = result;
	}

	printf("%llu\n",result);
}
