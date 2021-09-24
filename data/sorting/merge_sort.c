#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define SWAP(x,y) do {   \
		typeof(x) _tmp = x;      \
		x = y;                \
		y = _tmp;                \
	} while(0)

int isPow2(size_t n){
	return (n & (n-1)) == 0;
}

void mergeSort(int *A, const size_t N){
	size_t i = 1, j = 2, m;

	int count = 0;
	while(j <= N && ++count <= 5){
		printf("--------------------\nSTART\ti = %3d\tj = %3d\n", i, j);

		if(A[i-1] > A[j-1]){
			SWAP(A[i-1], A[j-1]);
		}

		// Print arr
		for(size_t i = 0;i < N;i++) printf(" %d", A[i]);
		printf("\n");

		// If multiple of 4th element, merge blocks of 2
		if((j % 4) == 0){
			// MERGE 4
			m = j;
			j = i - 2;

			// If block ordered
			if(A[j] < A[i-1]){
				j++; i++;
				goto skip;
			}else if(A[j-1] > A[i]){
				SWAP(A[j-1], A[i-1]);
				SWAP(A[j], A[i]);

				j++; i++;
				goto skip;
			}

			// Min of first goes first, next goes second
			SWAP(A[j], A[i-1]); // Put mins on left side
			if(A[j-1] > A[j]){
				SWAP(A[j-1], A[j]);
			}

			// Max
			if(A[i-1] > A[i]){
				SWAP(A[i-1], A[i]);
			}

			skip:
			j = m;

			// Print arr
			for(size_t i = 0;i < N;i++) printf(" %d", A[i]);
			printf("\n");
		}

		/*if(isPow2(i)){
			i = j + 1;
		}else{
			i += 1;
		}*/
		i = j + 1;
		j = i + 1;

		printf("END  \ti = %3d\tj = %3d\n", i, j);
	}

	printf("--------------------\nLAST\ti = %3d\tj = %3d\n", i, j);
}

int isSorted(int *arr, int len){
	for(int i = 0;i < len-1;i++) if(arr[i] > arr[i+1]) return 0;
	return 1;
}

int main(int argc, char *argv[]){
	srandom(time(NULL));

	// Get N
	size_t N = 10;
	if(argc == 2){
		int temp = atol(argv[1]);
		if(temp > 1) N = temp;
	}
	printf("N: %u\n", N);

	// Create array
	int *arr = malloc(N * sizeof(*arr));
	for(size_t i = 0;i < N;i++) arr[i] = random() % (2*N);
	/*arr[0] = 10;
	arr[1] = 9;
	arr[2] = 8;
	arr[3] = 7;
	arr[4] = 6;
	arr[5] = 5;
	arr[6] = 4;
	arr[7] = 3;
	arr[8] = 2;
	arr[9] = 1;*/

	printf("Array before:");
	for(size_t i = 0;i < N;i++) printf(" %d",arr[i]);
	printf("\n");

	// Sort
	mergeSort(arr, N);

	printf("Array after:");
	for(size_t i = 0;i < N;i++) printf(" %d",arr[i]);
	printf("\n");

	free(arr);
}
