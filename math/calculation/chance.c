#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main(){
	srand(time(0));// Seed random generator

	int N=10e6;

	float average=0.0;

	int attempts=30;
	for(int i=0;i<attempts;i++){
		int num = rand()%N + 1;

		int count=1;
		while(1){
			int tmp = rand()%N +1;
			if(tmp == num){
				average += count;
				break;
			}
			count++;
		}
	}

	average /= attempts;
	printf("Average number of guesses per %d runs: %.3f\n",attempts,average);
	printf("Percentage: %.4f\n",average/N);
}

