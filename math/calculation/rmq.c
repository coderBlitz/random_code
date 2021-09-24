#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<unistd.h>

double minVal(double *arr, long start, long end){
	if(arr == NULL || start < 0 || end < start) return NAN;

	double min = arr[start];
	for(long i = start;i <= end;i++) if(arr[i] < min) min = arr[i];

	return min;
}

double query(double **data, long i, long j){
	return data[i][j-i];
}

int main(int argc, char *argv[]){
	srand(time(0));

	long N = 200;
	if(argc >= 2){
		char *ptr = argv[1];
		long temp = strtol(argv[1], &ptr, 10);
		if(argv[1] != ptr) N = temp;
	}

	printf("Testing N = %ld\n", N);
	double *data= malloc(N * sizeof(*data));
	if(data == NULL){
		printf("malloc failed\n");
		return -1;
	}
	for(long i = 0;i < N;i++) data[i] = rand() / (double)RAND_MAX;

	double **mins = malloc(N * sizeof(*mins));
	if(mins == NULL){
		printf("malloc failed\n");
		return -1;
	}
	for(long i = 0;i < N;i++){
		mins[i] = malloc((N - i) * sizeof(**mins));
		if(mins == NULL){
			printf("malloc failed\n");
			goto cleanup;
		}

		mins[i][0] = data[i];
		for(long j = 1;j < N-i;j++){
			mins[i][j] = (mins[i][j-1] < data[i+j]) ? mins[i][j-1] : data[i+j];
		}
	}

	/*printf("Data:\n");
	for(long i = 0;i < N;i++) printf("%7.6lf ", data[i]);
	printf("\n");

	for(long i = 0;i < N;i++){
		for(long j = 0;j < N-i;j++){
			printf("%7.6lf ", mins[i][j]);
		}
		printf("\n");
	}*/

	long queries = 10;
	if(argc == 3){
		char *ptr = argv[2];
		long temp = strtol(argv[2], &ptr, 10);
		if(argv[2] != ptr) queries = temp;
	}

	printf("Verifying %ld queries..\n", queries);
	long s,t;
	double result, min;
	for(long i = 0;i < queries;i++){
		s = rand() % N;
		t = s + rand() % (N-s);
		result = minVal(data, s, t);
		min = query(mins, s, t);
		if(result != min){
			printf("min(%ld, %ld): %7.6lf != %7.6lf\n", s, t, min, result);
		}
	}


cleanup:
	for(long i = 0;i < N;i++) free(mins[i]);
	free(mins);
	free(data);

	return 0;
}
