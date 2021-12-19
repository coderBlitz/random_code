/* Tests the refresh rate of stdout, with rate being iterations per second.
//*/

#include<stdio.h>
#include<time.h>

int main(int argc, char *argv[]) {
	long counter = 0;
	long last = clock(), current;

	while(1){
		counter += 1;

		current = clock();

		printf("Rate = %8ld\r", counter);

		if((current - last) >= CLOCKS_PER_SEC){
			counter = 0;
			last = current;
			printf("\n");
		}
	}

    return 0;
}
