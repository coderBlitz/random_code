#include<stdio.h>
#include<stdlib.h>
#include<sys/random.h>
#include<time.h>
#include"btree.h"

int main(int argc, char *argv[]){
	long seed;
	getrandom(&seed, sizeof(seed), 0);
	srand(seed);
	long N = 10;
	struct btree bt = {3, 0, NULL}; // Degree X, size 0, NULL root pointer

	long tmpi;
	if(argc >= 2){
		 tmpi = strtol(argv[1], NULL, 10);
		if(tmpi >= 0) N = tmpi;
	}
	if(argc >= 3){
		tmpi = strtol(argv[2], NULL, 10);
		if(tmpi > 0) bt.degree = tmpi;
	}

	printf("Inserting %7ld data to b-tree with degree %3d\n", N, bt.degree);

	long val, dups = 0;
	long res;
	long *data = malloc(N * sizeof(*data));
	size_t size = 0;
	for(long i = 1;i <= N;i++){
		val = rand() % (8*N);
		//printf("Inserting %3d..\n", val);
		res = btree_insert(&bt, val);
		if(res){
			//printf("Insert FAILED: %d\n", res);
			dups++;
		}else{
			data[size++] = val;
		}
		//btree_print(&bt);
		//printf("after size = %3ld\n", bt.size);
	}

	printf("%d dups\n", dups);
	printf("Final tree size = %lu\n", bt.size);
	//btree_print(&bt);

	if((dups + bt.size) != N){
		printf("\nWARNING: Size mismatch. Expected %3d, got %3d\n", N, dups + bt.size);
	}

	// Validate existing data
	printf("Validating data..\n");
	for(long i = 0;i < size;i++){
		if(!btree_find(&bt, data[i])){
			printf("WARNING: data = %3ld not found!\n", data[i]);
		}
	}

	// Remove valid data
	printf("Removing data..\n");
	for(long i = 0;i < size;i++){
		val = data[i];
		printf("Removing %3d..\n", val);
		res = btree_delete(&bt, val);
		if(res){
			printf("Remove FAILED: %d\n", res);
		}
		//btree_print(&bt);
		//printf("after size = %3ld\n", bt.size);
	}
	printf("Final tree size = %lu\n", bt.size);

	// Clear
	btree_destroy(&bt);
	free(data);

	return 0;
}
