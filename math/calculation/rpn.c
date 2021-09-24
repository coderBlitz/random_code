/*	Reverse-Polish notation parser
Just a stack of values/operations
*/

#include<ctype.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

/**	Is-operator
	Returns num of required operands, -1 for invalid
**/
static inline int isop(char c){
	switch(c){
		case '+':
		case '-':
		case '*':
		case '/':
		case '^':
		case '&':
		case '|':
			return 2;
			break;
	}

	return 0;
}

static inline double calc(double *vals, long size, char *op){
	/* Check basic operations
	TODO: Add logical operations (?)
	*/
	switch(*op){
		case '+':
			return vals[size-2] + vals[size - 1];
			break;
		case '-':
			return vals[size-2] - vals[size - 1];
			break;
		case '*':
			return vals[size-2] * vals[size - 1];
			break;
		case '/':
			return vals[size-2] / vals[size - 1];
			break;
		case '^':
			return pow(vals[size-2], vals[size - 1]);
			break;
		case '&':
			return vals[size-2] & vals[size - 1];
			break;
		case '|':
			return vals[size-2] | vals[size - 1];
			break;
	}

	/* TODO: Add function checks for longer operations
		* sqrt
		* sin
		* cos
	*/
	return NAN;
}

int main(int argc, char *argv[]){
	char eq[64];
	//strcpy(eq, "10 2 + 3 - 4 / 5 7 * *");
	strcpy(eq, "3 4 2 * 1 5 - 2 3 ^ ^ / +");
	long len = strlen(eq);
	printf("%s\n", eq);

	long numCap = 512;
	double *nums = malloc(numCap * sizeof(*nums));
	if(nums == NULL){
		return -1;
	}
	long numSize = 0;

	/* Parse loop
	TODO: Parse 
	*/
	char *c = eq, *tmp;
	int res;
	double d;
	while(*c){
		if(isdigit(*c)){
			// Parse number
			d = strtod(c, &tmp);
			if(tmp != c){
				// Push number
				//printf("Parsed number = %lf\n", d);
				nums[numSize++] = d;
				c = tmp;

				if(numSize == numCap){
					numCap *= 2;
					double *temp = realloc(nums, numCap * sizeof(*nums));
					if(temp == NULL){
						return -2;
					}
					nums = temp;
				}
			}else{
				printf("parse error\n");
			}
		}else if(res = isop(*c)){
			// Pop top two numbers and calculate
			//printf("Found operator '%c'\n", *c);
			if(numSize < res){
				fprintf(stderr, "Missing %2d operands for '%c'.\n", res - numSize, *c);
				numSize = 0;
				break;
			}

			/*for(int i = 0;i < numSize;i++){
				printf("%lg ", nums[i]);
			}
			printf("\n");/*/

			d = calc(nums, numSize, c);
			//printf("size = %ld\tRes = %lg\n", numSize-1, d);
			nums[--numSize-1] = d;
		}

		c++;
	}

	if(numSize == 1){
		printf("%20.16lf\n", nums[0]); // Print result
	}else if(numSize > 1){
		fprintf(stderr, "Not enough operators given. Missing %3ld operations\n", numSize - 1);
	}else{
		//fprintf(stderr, "Something weird happened.\n");
	}

	free(nums);

	return 0;
}
