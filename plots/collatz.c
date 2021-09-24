#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

int main(){
	unsigned long long high = 100000000;// 1e9 or 16e8
	int *lens = malloc(high*sizeof(int));
	unsigned long long max=0;

	FILE *plot = popen("gnuplot -p","w");
	fprintf(plot, "set title 'C Collatz chain length'\n");
	fprintf(plot, "set terminal qt size 1280,720\n");

	fprintf(plot, "plot '-' every 3 with dot\n");
	for(unsigned long long i=0;i<high;i++){
		int count=0;
		unsigned long long num=i;
		while(num > 1){
			if(num%2 == 0) num /= 2;
			else num = num*3 + 1;
			count++;
		}
		lens[i] = count;
		if(lens[i] > lens[max]) max = i;
		fprintf(plot, "%lu %lu\n", i, count);
	}
	fprintf(plot, "e");
	fclose(plot);
	printf("Largest chain: %d (%llu)\n",lens[max],max);


	free(lens);
}
