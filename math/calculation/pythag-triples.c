/***
Brute force pythagorean triples.

Algo:
	Start with hypotenuse value c (c > 3)
	Compute (c^2 - (c-1)^2) == (c^2 - c^2 + 2c - 1) == (2c - 1).
	Minimum value to check is a = sqrt(2c - 1) (a nor b can be smaller, since the difference requires it)
	Check if (a) is a perfect square.
	Check if (c^2 - a^2) is square
	Report values if both true.
	Increment c and repeat.
***/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main(int argc, char *argv[]){
	unsigned int n = 300;
	unsigned int c = 2;
	unsigned int a, b, x1, x2;

	if(argc >= 2){
		n = strtoul(argv[1], NULL, 10);
	}

	while(++c <= n){
		//printf("c = %u\n", c);
		x1 = 2*c - 1;
		a = sqrt(x1);

		x2 = c*c - a*a;
		b = sqrt(x2);
		//printf("\ta = %u\tb = %u\n", a, b);
		for(;a <= b;a++){
			if(b*b == x2){
				// Triple (?)
				printf("(%u, %u, %u)\n", a, b, c);
			}

			x2 -= 2*a + 1; // Optimized by subtracting (2*a + 1) each time, instead of recomputing
			b = sqrt(x2);
			//printf("\ta = %u\tb = %u\n", a, b);
		}
	}

	return 0;
} 
