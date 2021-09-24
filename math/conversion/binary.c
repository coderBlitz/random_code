#include<stdio.h>
#include<stdlib.h>

int main(){
	unsigned long num;
	printf("Integer to Binary\nEnter the number to convert: ");
	scanf("%lu",&num);

	char string[33];// Will hold binary representation
	unsigned long mask = 0x1;
	for(int i=31;i>=0;i--){
		string[i] = (num & mask) ? '1':'0';
		mask *= 0x2;
	}
	string [32] = '\0';// To print correctly

	char start=0;
	for(int i=0;i<32;i++){
		if(start){
			printf("%c",string[i]);
			continue;
		}
		else if(!(start) && string[i]=='1'){ start = 1; i--; }
	}
	printf("\n");
}
