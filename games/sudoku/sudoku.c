#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<unistd.h>
#include<time.h>

#define BOARD_SIZE 9

char ** initBoard(){
	char **board = NULL;
	board = malloc(BOARD_SIZE * sizeof(*board));
	if(!board){
		return NULL;
	}
	for(uint8_t i = 0;i < BOARD_SIZE;i++){
		board[i] = malloc(BOARD_SIZE * sizeof(**board));
		if(!board[i]){
			fprintf(stderr, "Could not allocate board\n");
			for(uint8_t j = i-1;j >= 0;j--) free(board[i]);
			return NULL;
		}
		for(int j = 0;j < BOARD_SIZE;j++) board[i][j] = 0;
	}

	return board;
}

uint8_t get(char **const restrict board, const uint8_t row, const uint8_t col){
	if(board == NULL) return 0;
	if(row >= BOARD_SIZE || col >= BOARD_SIZE){
		fprintf(stderr, "Invalid row/col for get: (%u, %u)\n", row, col);
		return 0;
	}

	uint8_t val = board[row][col];
	if(val > BOARD_SIZE) return val-10;
	return val;
}

// Return true on value placement
uint8_t valid_placement(char **const restrict board, const uint8_t value, const uint8_t row, const uint8_t col){
	if(board == NULL) return 0;
	if(row >= BOARD_SIZE || col >= BOARD_SIZE){
		fprintf(stderr, "Invalid row/col for placement\n");
		return 1;
	}

	uint8_t set_row = row/3;
	uint8_t set_col = col/3;
	uint8_t set = (set_row*3) + set_col;

	uint8_t set_center_row = 1 + 3*set_row;
	uint8_t set_center_col = 1 + 3*set_col;

//	printf("(%u, %u) in set %u with center (%u, %u)\n", row, col, set, set_center_col, set_center_row);

	// Check row
	for(uint8_t i = 0;i < BOARD_SIZE;i++){
		if(i == col) continue;
//		if(board[row][i] == value) return 0;
		if(get(board, row, i) == value) return 0;
	}
	// Check column
	for(uint8_t i = 0;i < BOARD_SIZE;i++){
		if(i == row) continue;
//		if(board[i][col] == value) return 0;
		if(get(board, i, col) == value) return 0;
	}

	if(get(board, set_center_row-1, set_center_col-1) == value
		|| get(board, set_center_row-1, set_center_col) == value
		|| get(board, set_center_row-1, set_center_col+1) == value) return 0;
	if(get(board, set_center_row, set_center_col-1) == value
		|| get(board, set_center_row, set_center_col) == value
		|| get(board, set_center_row, set_center_col+1) == value) return 0;
	if(get(board, set_center_row+1, set_center_col-1) == value
		|| get(board, set_center_row+1, set_center_col) == value
		|| get(board, set_center_row+1, set_center_col+1) == value) return 0;

	return 1;
}

// Recursive brute-force solver
uint32_t solve(char ** restrict board, uint8_t row, uint8_t col){
	if(board == NULL){
		fprintf(stderr, "Board is NULL in solver\n");
		return 0;
	}
	/** Recursive (brute-force) solver **/
	/* Check every possible number for current position, recurse to next.
	   If the last possible number has been reached and is invalid, return
	   If current position is occupied, skip
	*/
	uint8_t next_col = col + 1;
	uint8_t next_row = (next_col >= BOARD_SIZE)?row+1:row;
	next_col = next_col % BOARD_SIZE;
	if(row >= BOARD_SIZE) return 1; // If at the end, should be solved

	if(board[row][col]) return solve(board, next_row, next_col); // If number exists
	else{
		for(uint8_t i = 1;i <= BOARD_SIZE;i++){
			if(valid_placement(board, i, row, col)){
				board[row][col] = i;
				uint8_t ret = solve(board, next_row, next_col);
				if(ret) return ret;
			}
		}
	}

	board[row][col] = 0; // Clear value to not screw up validation
	return 0;
}

void printBoard(char **const restrict board){
	if(board == NULL) return;
	for(uint8_t i = 0;i < BOARD_SIZE;i++){
		for(uint8_t j = 0;j < BOARD_SIZE;j++) printf("%d ", get(board, i, j));
		printf("\n");
	}
}

char ** parseCSV(char *const restrict filename){
	if(filename == NULL){
		fprintf(stderr, "Can't open nothing\n");
	}
	FILE *fp = fopen(filename, "ro");
	if(fp == NULL){
		fprintf(stderr, "Can't open file '%s'\n", filename);
		return NULL;
	}

	/** Initialize board **/
	char **board = NULL;

	board = initBoard();
	if(!board){
		fprintf(stderr, "Could not initialize board\n");
		return NULL;
	}

	uint8_t cur = 0;
	char c = 0, get = 1;
	for(uint8_t i = 0;i < BOARD_SIZE;i++){
		cur = 0;
		get = 1;
		while((c = fgetc(fp)) != EOF){
			if(c == '\n'){
				for(int j = cur;j < BOARD_SIZE;j++) board[i][j] = 0;
				break;
			}
			if(get){
				if(c > '9' || c < '0') board[i][cur] = 0;
				else board[i][cur] = c - '0';
				get = 0;
				cur++;
			}
			if(c == ',') get = 1;
		}
	}

	fclose(fp);
	return board;
}

char ** randomize(char ** restrict board){
	srand(time(0));

	if(board == NULL){
		board = initBoard();
		if(!board){
			return NULL;
		}
	}

	uint8_t spots = 20;
	uint8_t result = 0;
	uint8_t row = 0, col = 0;
	for(uint8_t i = 0;i < spots;i++){
		do{
			result = rand() % 81;
			row = result/BOARD_SIZE;
			col = result%BOARD_SIZE;
		}while(board[row][col]);

		do{
			result = rand() % BOARD_SIZE + 1;
		}while(!valid_placement(board, result, row, col));
		board[row][col] = result;
	}

	return board;
}

int main(int argc, char *argv[]){
	struct timespec start, end;

	/** Initialize board **/
	char **board = NULL;

	if(argc > 1){
		if(argv[1][0] == '-'){
			if(argv[1][1] == 'r'){
				board = randomize(NULL);
				if(!board){
					fprintf(stderr, "Randomize failed to create board\n");
					exit(1);
				}
			}
		}else{
			board = parseCSV(argv[1]);
			if(!board){
				fprintf(stderr, "Could not create board from file\n");
				exit(1);
			}
		}
	}else{
		board = initBoard();
		if(!board){
			fprintf(stderr, "Could not allocate board\n");
			exit(1);
		}
	}

	/** Call the solver **/
	printBoard(board);

	clock_gettime(CLOCK_REALTIME, &start);
	int ret = solve(board, 0, 0);
	clock_gettime(CLOCK_REALTIME, &end);

//	printf("Start: %lu.%09lds\nEnd: %lu.%09lds\n", start.tv_sec, start.tv_nsec, end.tv_sec, end.tv_nsec);
	end.tv_nsec -= start.tv_nsec;
	if(end.tv_nsec < 0) end.tv_nsec = ~end.tv_nsec + 1;
	end.tv_sec -= start.tv_sec;


	if(ret) printf("\nSolved in %lu.%09lds!\n", end.tv_sec, end.tv_nsec);
	else printf("\nCould not solve. Tried for %lu.%09lds\n", end.tv_sec, end.tv_nsec);
	printBoard(board);

	/** Cleanup **/
	for(uint8_t i = 0;i < BOARD_SIZE;i++) free(board[i]);
	free(board);
}
