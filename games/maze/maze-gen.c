/*	Maze generator.
	TODO: Implement randomized depth-first search algo

Cell array, 1 byte each, where 4 bits (a nibble) are boolean for adjacency to neighbor.
We define the bit order to be:
	0 (0x1) - North neighbor
	1 (0x2) - East neighbor
	2 (0x4) - South neighbor
	3 (0x8) - West neighbor
Could use high 4 bits to represent a wall (or something), if necessary.
*/

#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/random.h>

void print_maze(uint8_t *maze, long M, long N){
	const char vdiv = '_';
	const char hdiv = '|';
	const char gap = ' ';

	// Line buffer (1 chr per cell, plus 2 on either side)
	uint8_t buffer[3*N + 1];
	buffer[3*N] = 0;

	size_t idx;

	// Print top borders
	// Print middle dividers
	for(long j = 0;j < N;j++){
		idx = j;

		buffer[3*j] = vdiv;
		buffer[3*j + 1] = (maze[idx] & 0x1) ? gap : vdiv; // top wall
		buffer[3*j + 2] = vdiv;
	}
	printf("%s\n", buffer);

	// Print bulk of maze
	for(long i = 0;i < M;i++){
		// Print middle dividers
		for(long j = 0;j < N;j++){
			idx = i * N + j;

			buffer[3*j] = (maze[idx] & 0x8) ? vdiv : hdiv; // Left wall
			buffer[3*j + 1] = (maze[idx] & 0x4) ? gap : vdiv; // Bottom wall
			buffer[3*j + 2] = (maze[idx] & 0x2) ? vdiv : hdiv; // Right wall
		}
		printf("%s\n", buffer);
	}
}

// Fisher-Yates shuffle
static inline void shuffle(long *arr, const size_t len){
	long tmp, loc;
	for(size_t i = len-1;i > 0;i--){
		loc = random() % i; // Get pos

		// Swap
		tmp = arr[loc];
		arr[loc] = arr[i];
		arr[i] = tmp;
	}
}

/**	Generate maze using randomized depth-first search (DFS).
	
**/
long gen_maze(uint8_t *cells, const long M, const long N, const size_t start, size_t *stk){
	size_t cells_size = M * N;
	if(M < 2 || N < 2 || cells == NULL || start >= cells_size) return 1;

	size_t *stack;
	if(stk == NULL){
		stack = malloc(cells_size * sizeof(*stack));
		if(stack == NULL){
			return 1;
		}
	}else{
		stack = stk;
	}
	size_t sp = 0;

	/* Do depth search (assume maze is at least 2x2, though it shouldn't affect the code)
	*/
	stack[sp++] = 0; // Start in upper left corner

	size_t idx, next;
	uint8_t mask, back;
	long dirs[4] = {0x1, 0x2, 0x4, 0x8};
	long long_tmp;
	do{
		idx = stack[sp - 1]; // Get current cell
		//printf("Cur = %lu\n", idx);

		mask = 0x0F;
		if((idx % N) == 0) mask ^= 0x8; // If left side
		if((idx % N) == (N-1)) mask ^= 0x2; // If right side
		if(idx < N) mask ^= 0x1; // If top row
		if((idx / N) == (M-1)) mask ^= 0x4; // If bottom row

		// Loop over directions
		shuffle(dirs, 4);
		back = 1; // If going backwards
		for(int i = 0;i < 4;i++){
			long_tmp = dirs[i] & mask;

			// Set next
			switch(long_tmp){
			case 0x1:
				next = idx - N;
				//printf("Going up. %lu --> %lu\n", idx, next);
				break;
			case 0x2:
				next = idx + 1;
				//printf("Going right. %lu --> %lu\n", idx, next);
				break;
			case 0x4:
				next = idx + N;
				//printf("Going down. %lu --> %lu\n", idx, next);
				break;
			case 0x8:
				next = idx - 1;
				//printf("Going left. %lu --> %lu\n", idx, next);
				break;
			default:
				//printf("Funky direction %ld.\n", long_tmp);
				continue;
			}

			// Check if next is visited or not
			if(!cells[next]){
				stack[sp++] = next;
				cells[idx] |= long_tmp; // Set current cell
				back = 0;
				break;
			}
		}

		// Update next cell partially
		if(!back){
			switch(long_tmp){
			case 0x1:
				cells[next] = 0x4;
				break;
			case 0x2:
				cells[next] = 0x8;
				break;
			case 0x4:
				cells[next] = 0x1;
				break;
			case 0x8:
				cells[next] = 0x2;
				break;
			}
		}else{
			--sp;
		}
	}while(sp);

	if(stk == NULL){
		free(stack);
	}

	return 0;
}

int main(int argc, char *argv[]){
	long N = 5; // Width
	long M = N; // Height

	/* Get cmd arguments
	*/
	if(argc == 2){
		fprintf(stderr, "Please give both width and height for the maze.\n");
		return 1;
	}else if(argc == 3){
		M = strtoul(argv[1], NULL, 10);
		N = strtoul(argv[2], NULL, 10);
	}

	// Get and verify size
	const size_t cells_size = M * N;
	if(cells_size <= 0){
		fprintf(stderr, "Invalid dimensions: %ld x %ld\n", N, M);
		return 1;
	}

	// Alloc cells and stack
	uint8_t *cells = calloc(cells_size, sizeof(*cells));
	if(cells == NULL){
		return 1;
	}
	size_t *stack = malloc(cells_size * sizeof(*stack));
	if(stack == NULL){
		return 1;
	}

	// Seed random
	long long_tmp;
	getrandom(&long_tmp, sizeof(long_tmp), M*N);
	srandom(long_tmp);

	// Get maze of given size, starting from given cell
	gen_maze(cells, M, N, N-1, stack);

	// Open walls at start and end of maze
	cells[0] |= 0x1; // Remove top wall for this cell (maze entrance)
	cells[cells_size-1] |= 0x2; // Open right-side wall (maze exit)

	//printf("Completed:\n");

	print_maze(cells, M, N);

	if(memchr(cells, 0, cells_size)){
		printf("INVALID MAZE! Contains at least 1 isolated cell!!\n");
	}

	/* Cleanup
	*/
	free(stack);
	free(cells);

	return 0;
}
