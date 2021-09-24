#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

// 8KB / 7.81KiB per 1000
#define SIZE 2000 // Initial size of array for factors

/* Can more effeciently find factors for very large numbers
   After the square root of a number N, the factors start pairing
   with other factors smaller than the square root of N. So once
   you find all factors up to and including the square of N, just
   divide N by all previously found factors to get the remaining ones
*/
int main(int argc, char *argv[]){
	unsigned long long N = 18446744073709551615U;// Number to factor
	if(argc == 2){
		N = atoll(argv[1]);
	}
	int factors=0;// Count of the number of factors
	unsigned long long *nums = malloc(SIZE * sizeof(long long));// Array to hold factors

	for(long long i=1;i <= sqrt(N);i++){// Find all factors up to and including square
		if(N%i == 0){
			printf("%lld is a factor\n",i);
			nums[factors] = i;
			factors++;
		}
	}

	char skip=0;
	if((int)sqrt(N) * (int)sqrt(N) == N){// Perfect squares will have double factor in this
		skip=1;// This prevents the perfect squares from having idendtical factors in array
	}

printf("\nMath time!\n\n");

	int hold=factors;// Holds previous factor value for array, since 'factors' keeps increasing
	for(int i=skip;i<hold;i++){// Use simple math to get the rest of the factors
		nums[hold+i-skip] = N/nums[hold-i-1];// Fill up rest of array with factors in increasing order
		printf("%llu is a factor\n",nums[hold+i-skip]);
		factors++;
	}

//for(int i=0;i<factors;i++) printf("%lld\n",nums[i]);// Print all factors if you want to see that
	free(nums);// Free up array
	printf("%d factors\n",factors);
}
