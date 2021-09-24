/* Goal: Try and solve a linear system, i.e a matrix
*/

#include<stdio.h>
#include<stdlib.h>
#include<sys/random.h>
#include<math.h>
#include<time.h>

void printMatrix(double **matrix, const int ROWS, const int COLS){
	printf("[");
	for(int i = 0;i < ROWS;i++){
		printf("%12.8lf", matrix[i][0]);
		for(int j = 1;j < COLS;j++){
			printf(" %12.8lf", matrix[i][j]);
		}
		printf("\n");
	}
	printf("]\n");
}

void swapIntRows(int **matrix, const int COLS, const int ROW1, const int ROW2){
	for(int i = 0;i < COLS;i++){// Swap rows
//		printf("Swapping %d and %d..\n", matrix[ROW1][i], matrix[ROW2][i]);
		matrix[ROW1][i] ^= matrix[ROW2][i];// Row one holds (a XOR b)
		matrix[ROW2][i] ^= matrix[ROW1][i];// Row two holds b XOR (a XOR b) == a
		matrix[ROW1][i] ^= matrix[ROW2][i];// Row one holds (a XOR b) XOR a == b
	}
}

void swapRows(double **matrix, const int COLS, const int ROW1, const int ROW2){
	double temp = 0;
	for(int i = 0;i < COLS;i++){// Swap rows
		temp = matrix[ROW1][i];
		matrix[ROW1][i] = matrix[ROW2][i];
		matrix[ROW2][i] = temp;
	}
}

/**	Gaussian elimination/reduction of matrix
**/
#define EPS 0.0000000000000001
void solve(double **matrix, const int ROWS, const int COLS){
	double max;
	int max_row;

	int rank = 0;
	for(int piv = 0;piv < COLS;piv++){
		//printf("piv = %2d\n", piv);

		// Find largest pivot to reduce error
		max = matrix[rank][piv];
		max_row = rank;
		for(int i = rank;i < ROWS;i++){
			if(fabs(matrix[i][piv]) > fabs(max)){
				max = matrix[i][piv];
				max_row = i;
			}
		}

		// Determine if no pivot in this column. Can use small epsilon if results weird
		//printf("Max_row = %2d\tmax = %lf\n", max_row, max);
		if(fabs(max) <= EPS){
			//printf("Max value is nearly 0\n");
			continue;
		}

		// Swap largest pivot to next pivot row
		if(max_row != rank){
			//printf("Swapping row %d with %d\n", rank, max_row);
			swapRows(matrix, COLS, max_row, rank);
		}

		// Scale remaining entries in pivot row
		//printf("Scaling row %2d by %lf\n", rank, max);
		for(int j = piv;j < COLS;j++){
			matrix[rank][j] /= max;
		}

		//printMatrix(matrix, ROWS, COLS);

		// Go through all rows and reduce using pivot (which should be 1)
		double scale;
		for(int i = 0;i < ROWS;i++){
			scale = matrix[i][piv];
			if(fabs(scale) <= EPS || i == rank) continue; // Skip pivot row itself

			// Scale the remaining row entries
			//printf("Reducing row %2d with scalar %lf\n", i, scale);
			for(int j = piv;j < COLS;j++){
				//printf("%lf -= %lf * %lf\n", matrix[i][j], scale, matrix[rank][j]);
				matrix[i][j] -= scale * matrix[rank][j];
			}
		}

		rank += 1;
		if(rank == ROWS) break;
		//printMatrix(matrix, ROWS, COLS);
	}

	printf("Rank = %d\n", rank);
}

int main(){
	long seed;
	getrandom(&seed, sizeof(seed), 0);
	srand(seed);
	int ROWS = 3,COLS = 4;// ROWxCOL size matrix
	double **matrix = calloc(ROWS, sizeof(double *));
	for(int i = 0;i < ROWS;i++) matrix[i] = calloc(COLS, sizeof(double));

	for(int i = 0;i < ROWS;i++){
		for(int j = 0;j < COLS;j++){
			//matrix[i][j] = rand() % 100;
			matrix[i][j] = (double) rand() / (double) RAND_MAX;
			//matrix[i][j] = 0;
		}
	}

	/*matrix[0][2] = 2.0;
	matrix[0][3] = 1.0;
	matrix[1][2] = 3.0;
	matrix[1][3] = 5.0;*/

	printMatrix(matrix, ROWS, COLS);
	solve(matrix, ROWS, COLS);
	printMatrix(matrix, ROWS, COLS);

	for(int i = 0;i < ROWS;i++) free(matrix[i]);
	free(matrix);
}
