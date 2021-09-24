#ifndef QUICKSORT_H_
#define QUICKSORT_H_

#include<stdlib.h>

int partition(int *arr, int start, int end){
	// Pick pivot
	int pivot = arr[end];

	// Partition
	int temp;
	int j = start; // j is first index of higher partition
	for(int i = start;i < end;i++){
		if(arr[i] < pivot){
			// Swap values
			temp = arr[i];
			arr[i] = arr[j];
			arr[j] = temp;
			j++;
		}
	}

	// Swap pivot with first element of higher partition, or last of lower
	temp = arr[end];
	arr[end] = arr[j];
	arr[j] = temp;

	return j;
}

int partition_rand(int *arr, int start, int end){
	int pos = rand() % (end-start) + start;
	int temp = arr[pos];
	arr[pos] = arr[end];
	arr[end] = temp;
	return partition(arr, start, end);
}

void quicksort(int *arr, int start, int end){
	if(start < 0 || end < 0) return;
	if(end - start < 1) return;

	int j = partition(arr, start, end);

	// Repeat
	quicksort(arr, start, j - 1);
	quicksort(arr, j+1, end);
}

#endif
