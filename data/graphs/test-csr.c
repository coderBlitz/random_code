#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"csr.h"
#include"quicksort.h"

long checkSizes(struct csr_graph const *graph, size_t *insert_count, size_t N, unsigned int offset){
	printf("Checking row size and columns..\n");

	size_t max_count = 0;
	unsigned int max_vert_diff = N - graph->num_rows + offset;
	for(int i = 0;i < N-max_vert_diff;i++){
		if(insert_count[i] > max_count) max_count = insert_count[i];

		// Check size
		size_t row_size = graph->rows[i+1 + offset] - graph->rows[i + offset];
		if(row_size != insert_count[i]){
			printf("SIZE MISMATCH FOR ROW %u! Got: %u\texpected: %u\n", i, row_size, insert_count[i]);
			printf("INDICES %2u and %2u\n", i+1+offset, i+offset);
		}
	}
	printf("Check done. Max count: %u\n", max_count);
	//dumpGraph(&gr);

	// Check NNZ entry at end of row array
	if(graph->rows[N + offset - max_vert_diff] != N){
		printf("SIZE MISMATCH! Got: %u\t(%u)\texpected: %u\n", graph->rows[N], graph->rows[N + offset], N);
		/*for(size_t i = 0;i <= graph->num_rows;i++){
			printf("%u: %u\n", i, graph->rows[i]);
		}
		printf("\n");*/
	}
}

void checkData(struct csr_graph const *graph, size_t *insert_count, double **edge_check, size_t N, unsigned int offset){
	unsigned int max_vert_diff = N - graph->num_rows+1;
	printf("Num_rows: %u\tdiff: %u\n", graph->num_rows, max_vert_diff);

	for(int i = 0;i < N-max_vert_diff;i++){
		// Check data validity
		size_t row_start = graph->rows[i + offset];

		for(int j = 0;j < insert_count[i];j++){
			unsigned int col = graph->cols[row_start+j] - offset;

			if(graph->vals[row_start+j] != edge_check[i][col]){
				printf("VAL MISMATCH FOR (%u, %u). Got: %u\texpected: %u\n", i, col, graph->vals[row_start+j], edge_check[i][col]);
			}
		}
	}
	//dumpGraph(&gr);
}

int insertTest(const size_t N){
	struct csr_graph gr;
	initGraph(&gr);

	// Will be dense matrix that stores same information as CSR
	double **edge_check = malloc(N * sizeof(*edge_check));
	if(edge_check == NULL){
		fprintf(stderr, "No mems\n");
		return -1;
	}
	for(int i = 0;i < N;i++){
		edge_check[i] = malloc(N * sizeof(**edge_check));
		if(edge_check[i] == NULL){
			fprintf(stderr, "No mems again\n");
			return -1;
		}
	}

	// Count the size of each row during insert
	size_t simple_count[N];
	for(int i = 0;i < N;i++) simple_count[i] = 0;

	// Insert random vertex pairings, in range [offset, N + offset)
	int offset = 1;
	unsigned int dup_count = 0;
	for(int i = 0;i < N;i++){
		int a = rand() % N + offset;
		int b = rand() % N + offset;

		//printf("Inserting (%d, %d)\n", a, b);
		int ret = insertArc(&gr, a, b, a*b/2.0);
		//dumpGraph(&gr);
		if(ret < 0){
			dup_count++;
			i--;
			continue;
		}

		// Truth data
		edge_check[a-offset][b-offset] = a*b/2.0;
		simple_count[a - offset]++;
	}
	printf("%u duplicates encountered.\n", dup_count);

	//dumpGraph(&gr);

	/*	Make sure that each row is the correct size, and that the row has the
		 correct data in each column.
	*/
	checkSizes(&gr, simple_count, N, offset);
	checkData(&gr, simple_count, edge_check, N, offset);

	destroyGraph(&gr);

	for(int i = 0;i < N;i++){
		free(edge_check[i]);
	}
	free(edge_check);
}

int main(int argc, char *argv[]){
	srand(time(0));

	const size_t N = 10;

	struct csr_graph gr;
	initGraph(&gr);

	// Count the size of each row during insert
	size_t simple_count[N];
	for(int i = 0;i < N;i++) simple_count[i] = 0;

	// Insert random vertex pairings, in range [offset, N + offset)
	unsigned int dup_count = 0;
	for(int i = 0;i < N;i++){
		int a = rand() % N;
		int b = rand() % N;

		int ret = insertArc(&gr, a, b, a*b/2.0);
		if(ret < 0){
			dup_count++;
			i--;
			continue;
		}

		simple_count[a]++;
	}
	printf("%u duplicates encountered.\n", dup_count);

	checkSizes(&gr, simple_count, N, 0);
	dumpGraph(&gr);

	size_t cnt = 0;
	unsigned int *temp;
	for(int i = 0;i < N;i++){
		temp = getNeighbors(&gr, i, &cnt);
		if(!cnt) continue;

		for(int j = 0;cnt;++j){
			printf("Removing (%u,%u)\n", i, temp[0]);
			removeArc(&gr, i, temp[0]);
			temp = getNeighbors(&gr, i, &cnt);
		}

		dumpGraph(&gr);
	}

	destroyGraph(&gr);

	return 0;
}
