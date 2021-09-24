#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define SWAP(x,y) do {   \
		typeof(x) _tmp = x;      \
		x = y;                \
		y = _tmp;                \
	} while(0)

//int partition(int *arr, int start, int end){
int partition(int *restrict const arr, int i, const int end){
	// Pick pivot
	const int pivot = arr[end];

	// Partition
	int temp;
	int j = i; // j is first index of higher partition
	for(;i < end;++i){
		if(arr[i] < pivot){
			// Swap values
			SWAP(arr[i], arr[j]);
			++j;
		}
	}

	// Swap pivot with first element of higher partition, or last of lower
	SWAP(arr[end], arr[j]);

	return j;
}

int partition_rand(int *const arr, const int start, const int end){
	int pos = rand() % (end-start) + start;
	int temp = arr[pos];
	arr[pos] = arr[end];
	arr[end] = temp;
	return partition(arr, start, end);
}

void quicksort(int *restrict const arr, const int start, const int end){
	if(start < 0 || end < 0 || (end - start) < 1) return;

	const int j = partition(arr, start, end);

	// Repeat
	quicksort(arr, start, j - 1);
	quicksort(arr, j+1, end);
}

void quicksort_iter(int *arr, int l, int h){
	// Use stack to push start and end bounds
	// While stack isn't empty
// Create an auxiliary stack 
	int stack[h - l + 1]; 
  
	// initialize top of stack 
	int top = -1; 
  
	// push initial values of l and h to stack 
	stack[++top] = l; 
	stack[++top] = h; 
  
	// Keep popping from stack while is not empty 
	while (top >= 0) { 
		// Pop h and l 
		h = stack[top--]; 
		l = stack[top--]; 
  
		// Set pivot element at its correct position 
		// in sorted array
		int p = partition(arr, l, h);
  
		// If there are elements on left side of pivot, 
		// then push left side to stack 
		if (p - 1 > l) { 
			stack[++top] = l; 
			stack[++top] = p - 1; 
		} 
  
		// If there are elements on right side of pivot, 
		// then push right side to stack 
		if (p + 1 < h) { 
			stack[++top] = p + 1; 
			stack[++top] = h; 
		} 
	} 
}

int isSorted(int *arr, int len){
	for(int i = 0;i < len-1;i++) if(arr[i] > arr[i+1]) return 0;
	return 1;
}

void timespecsub(struct timespec *a, struct timespec *b, struct timespec *res){
	res->tv_sec = a->tv_sec - b->tv_sec;
	res->tv_nsec = a->tv_nsec - b->tv_nsec;
	if(res->tv_nsec < 0){
		res->tv_nsec += 1000000000;
		res->tv_sec -= 1;
	}
}

int compare_int(const void *a, const void *b){
	int ret = (*(int *)a < *(int *)b) ? -1 : 0;
	ret = (*(int *)a > *(int *)b) ? 1 : 0;
	return ret;
}

int main(int argc, char *argv[]){
	srandom(time(0));

	uint32_t N = 1000;
	if(argc == 2){
		int temp = atol(argv[1]);
		if(temp > 0) N = temp;
	}
	printf("N: %u\n", N);

	struct timespec start, end, diff;

	//int A[] = {6,24,15,2,28,18,23,7,18,10};
	int *A = malloc(N * sizeof(*A));
	for(uint32_t i = 0;i < N;i++) A[i] = random() % N;

	//for(int i = 0;i < N;i++) printf("%d%s", A[i], ((i == N-1)?"\n":", "));
	printf("Is sorted: %d\n", isSorted(A, N));

	clock_gettime(CLOCK_MONOTONIC, &start);
	quicksort(A, 0, N-1);
	//quicksort_iter(A, 0, N-1);
	//qsort(A, N, sizeof(*A), compare_int);
	clock_gettime(CLOCK_MONOTONIC, &end);

	timespecsub(&end, &start, &diff);

	//for(int i = 0;i < N;i++) printf("%d%s", A[i], ((i == N-1)?"\n":", "));
	printf("Is sorted: %d\n", isSorted(A, N));

	printf("Time: %lu.%09lu\n", diff.tv_sec, diff.tv_nsec);

	free(A);
}
