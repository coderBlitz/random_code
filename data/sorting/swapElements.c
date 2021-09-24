#include<stdio.h>

void swap(int *array,int pos1,int pos2){// Switch array elements at pos1 and pos2
	int tmp = array[pos1];
	array[pos1] = array[pos2];
	array[pos2] = tmp;
}

int main(){
	int SIZE = 10;
	int arr[SIZE];
	for(int i=0;i<SIZE;i++) arr[i] = i;

	printf("arr:");
	for(int i=0;i<SIZE;i++) printf(" %d",arr[i]);
	printf("\n");

	swap(arr,2,4);

	printf("new arr:");
	for(int i=0;i<SIZE;i++) printf(" %d",arr[i]);
	printf("\n");
}
