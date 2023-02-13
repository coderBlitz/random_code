#include<stdio.h>
#include<stdlib.h>

int main(int argc, char *argv[]){
	unsigned long high = 100000000;
	if(argc >= 2){
		high = strtol(argv[1], NULL, 10);
	}

	short *lens = malloc(high * sizeof(*lens));
	unsigned long max = 0;

	for(unsigned long i = 2;i < high;i++){
		short count = 0;
		unsigned long num = i;
		while(num > 1){
			if((num % 2) == 0) num /= 2;
			else num = num * 3 + 1;
			count++;

			if(num < i){
				// Stop counting, cause we've already done it
				count += lens[num];
				break;
			}
		}
		lens[i] = count;
		if(lens[i] > lens[max]) max = i;
	}
	printf("Largest chain: %d (%llu)\n", lens[max], max);


	free(lens);
}
