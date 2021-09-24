#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>

struct n_matrix{
	uint32_t rows;
	uint32_t cols;
	double *data;
};

void print_matrix(struct n_matrix A){
	// Print loop
	printf("[\n");
	for(int i = 0;i < A.rows;++i){
		printf("  [%lf", A.data[i*A.cols]);
		for(int j = 1;j < A.cols;++j) printf(", %lf", A.data[i*A.cols + j]);
		printf("]\n");
	}
	printf("]\n");
}

struct n_matrix multiply(struct n_matrix A, struct n_matrix B){
	struct n_matrix C = {A.rows, B.cols, NULL};

	if(A.cols != B.rows){
		printf("Incompatible sizes\n");
		return C;
	}

	C.data = malloc(C.rows*C.cols * sizeof(*C.data));
	if(C.data == NULL){
		printf("Could not allocate matrix for result\n");
		return C;
	}

	// Calculation loop
	for(int i = 0;i < A.rows;++i){
		// For each row in A
		for(int j = 0;j < B.cols;++j){
			// For each column in B
			double sum = 0;
			for(int k = 0;k < A.cols;++k) sum += A.data[i*A.cols + k]*B.data[k*B.cols + j];
			C.data[i*A.rows + j] = sum;
		}
	}

	return C;
}

int basic_solve(struct n_matrix *A){
	// Read math book, then maybe do LU-matrix

	return 0;
}

int main(int argc, char *argv[]){
	struct n_matrix A = {1,3,NULL};
	struct n_matrix B = {3,1,NULL};

	A.data = malloc(A.rows*A.cols * sizeof(*A.data));
	if(A.data == NULL){
		printf("Could not allocate matrix for A\n");
		return -1;
	}
	B.data = malloc(B.rows*B.cols * sizeof(*B.data));
	if(A.data == NULL){
		printf("Could not allocate matrix for B\n");
		return -1;
	}

	A.data[0] = 1;
	A.data[1] = 2;
	A.data[2] = 3;

	B.data[0] = 4;
	B.data[1] = 5;
	B.data[2] = 6;

	print_matrix(A);
	print_matrix(B);

	printf("=\n");
	struct n_matrix C = multiply(A, B);
	print_matrix(C);

	free(C.data);
	free(B.data);
	free(A.data);
	return 0;
}
