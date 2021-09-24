#include<stdio.h>
#include<stdlib.h>

// Game of Chance
// Picks a ramdom number in range N, then uses random num gen
// to guess it. Counts number of guesses to get to number

int main(){
	srand(time(0));// Seed the rand generator
	int N = 10e6;// 10e6 = 10,000,000
	printf("MAX: %d\n",N);

	int num = rand()%N + 1;
	printf("Looking for number: %d...\n",num);

	int count=1;
	while(1){
		int tmp = rand()%N + 1;
		if(tmp == num){
		   printf("Found after %d attempts\n",count);
		   printf("Percent of all numbers guessed: %f\n",(float)count/N);
		   break;
		}
		count++;
	}
}