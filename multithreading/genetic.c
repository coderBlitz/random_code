#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<time.h>
#include<pthread.h>
 
const char target[] = "Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battlefield of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. But, in a larger sense, we can not dedicate, we can not consecrate, we can not hallow this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us—that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion—that we here highly resolve that these dead shall not have died in vain—that this nation, under God, shall have a new birth of freedom—and that government of the people, by the people, for the people, shall not perish from the earth.";// Target string

const char tbl[] = ",.-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ";

#define CHOICE (sizeof(tbl) - 1) // Used for random number or something
#define MUTATE 30 // Probably degree of mutation
#define COPIES 256 // Probably number of children spawned

char specimen[COPIES+1][sizeof(target) / sizeof(char)];
 
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
	register int i, sum = 0;
	for (i = 0; a[i]; i++)
		sum += (a[i] != b[i]);
	return sum;
}
 
/* each char of b has 1/MUTATE chance of differing from a */
void mutate(int id) // Convert to GPU code along with irand
{
	int ID = id;
	register int i;
	for (i = 0; specimen[0][i]; i++)
		specimen[ID][i] = irand(MUTATE) ? specimen[0][i] : tbl[irand(CHOICE)];
 
	specimen[ID][i] = '\0';
}
 
int main()
{
	srand(time(0));

	register int i, best;
	int best_i, unfit, iters = 0;


	/* init rand string */
	for (i = 0; target[i]; i++)
		specimen[0][i] = tbl[irand(CHOICE)];
	specimen[0][i] = 0;
 
	void (*mutatePtr) = mutate;
	do { // MAIN PROGRAM LOOP
		pthread_t pth[COPIES];
		for (i = 1; i <= COPIES; i++){ // Loop that spawns children. Replace with GPU function code
			pthread_create(&pth[i-1], NULL, mutatePtr, (void *)(long)(i));
//			printf("I: %d\n",i);
		}
		for(i=0;i<COPIES;i++){
			pthread_join(pth[i],NULL);
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
		if(iters % 10 == 0){
			printf("\riter %d, score %d", iters, best);
			fflush(stdout);
		}
		iters++;
	} while (best);
 
printf("\n");
printf("Final string in %d iterations. '%s'\n",iters,specimen[0]);
	return 0;
}
