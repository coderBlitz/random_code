#include<stdio.h>
#include<stdlib.h>
#include<time.h>

// Sorts numbers in 'array' of length 'SIZE' is descending order
void sortDesc(int *const array,const int SIZE){
	for(int i=0;i<SIZE-1;i++){
		if(array[i] >= array[i+1]) continue;
		else{
			int tmp = array[i];
			array[i] = array[i+1];
			array[i+1] = tmp;
			i=-1;
		}
	}
}

// Sorts numbers in 'array' of length 'SIZE' is ascending order
void sortAsc(int *const array,const int SIZE){
	for(int i=0;i<SIZE-1;i++){
		if(array[i] <= array[i+1]) continue;
		else{
			int tmp = array[i];
			array[i] = array[i+1];
			array[i+1] = tmp;
			i=-1;
		}
	}
}

int main(){
	srandom(time(NULL));
	int SIZE = 15;
	int arr[SIZE];
	for(int i=0;i<SIZE;i++) arr[i] = random() % 51;

	printf("Array before:");
	for(int i=0;i<SIZE;i++) printf(" %d",arr[i]);
	printf("\n");

	sortAsc(arr,SIZE);// sortAsc(array,arraySize);

	printf("Array after:");
	for(int i=0;i<SIZE;i++) printf(" %d",arr[i]);
	printf("\n");
}
