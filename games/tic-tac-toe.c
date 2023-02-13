#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<time.h>

static char board[4][4];
static char turn = 1;

void printBoard(){
	printf("-------------\n");
	for(int i=0;i<3;i++){
		printf("|");
		for(int n=0;n<3;n++){
			switch(board[i][n]){
			  case 0: printf(" %d ",(n + i*3)+1);
					break;
			  case 1: printf(" X ");// Player will be 'X'
					break;
			  case 2: printf(" O ");// Computer will be 'O'
					break;
			}
			printf("|");
		}
		printf("\n-------------\n");
	}
}

void computer(){// Computer is always 2
	if(turn){
		char pick = 0;
		do{
			pick = rand() % 9;
		} while(board[pick/3][pick%3] != 0);
		board[pick/3][pick%3] = 2;
	}
}

int check_game(void){
	printf("%08x\n", *(int *)board);
	printf("%08x\n", *(int *)(board + 1));
	printf("%08x\n", *(int *)(board + 2));
}

int main(){
	srand(time(0));
	for(int i=0;i<9;i++) board[i/3][i%3] = rand() % 3;// Random garbage values to test print

	printBoard();
	check_game();
}
