#include"sorted_vec.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main(){
	srand(time(0));
	int N = 10;

	SortedBuffer test = newBuffer(N);
	int64_t data[N];

	for(int i = 0;i < N;++i){
		data[i] = rand() % 100;
//		printf("Inserting %ld\n", data[i]);
		insertBuffer(test, data[i]);
	}
	printBuffer(test);

	for(int i = 0;i < N;++i){
		if(searchBuffer(test, data[i]) < 0){
			printf("Could not find %ld in buffer\n", data[i]);
		}
	}

	freeBuffer(test);
}
