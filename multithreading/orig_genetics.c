#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<time.h>
 
const char target[] = "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal";// Target string
const char tbl[] = ",.-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ";
 
#define CHOICE (sizeof(tbl) - 1) // Used for random number or something
#define MUTATE 15 // Probably degree of mutation
#define COPIES 100 // Probably number of children spawned
 
/* returns random integer from 0 to n - 1 */
int irand(int n) //Convert to GPU code along with mutate
{
	int r, rand_max = RAND_MAX - (RAND_MAX % n);
	while((r = rand()) >= rand_max);
	return r / (rand_max / n);
}
 
/* number of different chars between a and b */
int unfitness(const char *a, const char *b)
{
	int i, sum = 0;
	for (i = 0; a[i]; i++)
		sum += (a[i] != b[i]);
	return sum;
}
 
/* each char of b has 1/MUTATE chance of differing from a */
void mutate(const char *a, char *b) // Convert to GPU code along with irand
{
	int i;
	for (i = 0; a[i]; i++)
		b[i] = irand(MUTATE) ? a[i] : tbl[irand(CHOICE)];
 
	b[i] = '\0';
}
 
int main()
{
	srand(time(0));
printf("Using %d children\n",COPIES);

	int i, best_i, unfit, best, iters = 0;
	char specimen[COPIES][sizeof(target) / sizeof(char)];
 
	/* init rand string */
	for (i = 0; target[i]; i++)
		specimen[0][i] = tbl[irand(CHOICE)];
	specimen[0][i] = 0;
 
	do { // MAIN PROGRAM LOOP
		for (i = 1; i < COPIES; i++){ // Loop that spawns children. Replace with GPU function code
			mutate(specimen[0], specimen[i]);
		}
 
		/* find best fitting string */
		for (best_i = i = 0; i < COPIES; i++) {
			unfit = unfitness(target, specimen[i]);
			if(unfit < best || !i) {
				best = unfit;
				best_i = i;
			}
		}
 
		if (best_i) strcpy(specimen[0], specimen[best_i]);
		printf("\riter %d, score %d: ", iters++, best);
	} while (best);
 
printf("\n");
	return 0;
}
