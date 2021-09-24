#include<stdio.h>
#include<stdlib.h>

int main(){
	int tmp;
	printf("Find number at N in fibonacci series: ");
	scanf("%d",&tmp);

	if(tmp == 1 || tmp == 2){
		printf("1\n");
		exit(0);
	}

	long first=1,second=1,num;
	for(int i=3;i<=tmp;i++){
		num = first + second;
		first = second;
		second = num;
	}
	printf("%ld\n",num);
}
