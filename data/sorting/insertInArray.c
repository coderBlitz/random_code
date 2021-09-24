#include<stdio.h>
#include<stdlib.h>

void insert(int *array,int *SIZE,int value){// Add 'value' into array of size 'SIZE'
	array = realloc(array,(*SIZE+1) * sizeof(int));
	if(array == NULL){
		printf("INSERT_ERROR: Could not resize array to fit new element\n");
		return;
	}

	(*SIZE)++;// Update with new size

	int tmp = array[0];
	array[0] = value;
	int tmp2;
	for(int n=1;n<*SIZE;n++){
		tmp2 = array[n];
		array[n] = tmp;
		tmp = tmp2;
	}
}

int main(){
	int SIZE = 10;
	int *arr = malloc(SIZE * sizeof(int));
	for(int i=0;i<SIZE;i++) arr[i] = i;

	printf("arr:");
	for(int i=0;i<SIZE;i++) printf(" %d",arr[i]);
	printf("\n");

	insert(arr,&SIZE,-1);

	printf("new arr:");
	for(int i=0;i<SIZE;i++) printf(" %d",arr[i]);
	printf("\n");

	free(arr);
}
