/***	pollard.c -- Pollard factoring variants
	Author: Chris Skane
	Date: 30 APR 2020
	Build: Use included Makefile, or simply run `gcc -o pollard.out pollard.c -lgmp -lm`
	Notes:

	Command-line options:
		Usage: ./pollard <method> [options] NUM
		Method:
			'-R' -- Rho, Use the Pollard-rho method
			'-P' -- P-1, Use the Pollard p-1 method
			'-B' -- Brent, Use the Pollard-brent method
		Options:
			'-b' -- p-1 B, the B value used for Pollard p-1 (default 30)
			'-c' -- Brent constant, the constant used in the Pollar-brent equation (default 1)
***/

#include<errno.h>
#include<gmp.h>
#include<math.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/random.h>
#include<time.h>
#include<unistd.h>

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

// Primes used if not generating primes
/*static const uint64_t primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241,
251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353,
359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461,
463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587,
593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691,
701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823,
827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947,
953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051,
1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279,
1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399,
1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489,
1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601,
1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709,
1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831,
1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951,
1973, 1979, 1987, 1993, 1997, 1999};*/
static uint64_t *primes;

static uint64_t MAX_B = 30;

/*	Find all primes up to and including max
*/
size_t gen_primes(size_t max, uint64_t **arr){
	char *primes = malloc(max * sizeof(*primes));
	if(primes == NULL){
		perror("Could not allocate sieve");
		exit(1);
	}
	for(unsigned long i = 0;i < max;i++) primes[i] = 1;

	if(*arr == NULL){
		*arr = malloc(max * sizeof(**arr));
		if(*arr == NULL){
			perror("Could not allocate results");
			exit(1);
		}
	}

	size_t count = 0;
	for(uint64_t i = 2; i < max; i++){
		if(primes[i] == 0) continue;
		else{
			(*arr)[count] = i;
			++count;
		}
		if(i*i < max){
			for(uint64_t n = i * 2; n < max; n += i){
				primes[n] = 0;
			}
		}
	}

	return count;
}

/*	Pollard-rho algorithm, for given `num`
*/
long rho(const mpz_t num, gmp_randstate_t state){
	mpz_t xi, y, x, d; // x is intermediate calculation, d is gcd
	mpz_inits(xi, y, x, d, 0);

	uint64_t i = 1, k = 2; // Normal types because these will NEVER overflow in any practical time
	mpz_urandomm(xi, state, num);
	mpz_set(y, xi);

	while(1){
		++i;

		// xi = (xi^2 - 1)(mod N)
		mpz_powm_ui(xi, xi, 2, num);
		mpz_sub_ui(xi, xi, 1);
		if(mpz_sgn(xi) < 0) mpz_add(xi, xi, num);

		mpz_sub(x, y, xi);
		mpz_gcd(d, x, num);

		if(mpz_cmp_ui(d, 1) && mpz_cmp(d, num)){
			gmp_printf("Factor: %Zd\n", d);
			break;
		}

		if(i == k){
			mpz_set(y, xi);
			k *= 2;
		}
	}

	mpz_clears(xi, y, x, d, 0);

	return 0;
}

/*	Pollard p-1 algorithm
*/
long p1(const mpz_t num){
	if(primes == NULL){
		fprintf(stderr, "P1 failure: Primes array is empty.\n");
		return -1;
	}

	mpz_t M, a, d;
	mpz_inits(M, a, d, 0);

	mpz_root(a, num, 6);
	gmp_printf("Ideal B: %Zd\n", a);
	uint64_t start;

	// Generate primes
	if(mpz_cmp_ui(a, 10000) < 0){
		start = 2;

		free(primes);
		primes = NULL;
		MAX_B = mpz_get_ui(a) + 20000;
		printf("Using B = %lu instead\n", MAX_B);
		gen_primes(MAX_B, &primes);
	}else{
		start = MAX_B-2;
	}

	//uint64_t B = 3;
	mpz_set_ui(a, 2);

	for(uint64_t B = start;B < MAX_B;++B){
		// Calculating M
		mpz_set_ui(M, 1);
		int ex;
		for(int i = 0;primes[i] <= B;i++){
			// d = p^ex
			ex = floor(log(B)/log(primes[i]));
			mpz_set_ui(d, primes[i]);
			mpz_pow_ui(d, d, ex);

			mpz_mul(M, M, d);
		}
		//gmp_printf("B: %lu\nM: %Zd\n", B, M);

		// M now holds result of exponentiation, since it is not needed afterwards
		mpz_powm(M, a, M, num);
		mpz_sub_ui(M, M, 1);
		if(mpz_sgn(M) < 0) mpz_add(M, M, num);

		// Using wiki check out of curiosity
		mpz_gcd(d, M, num);
		/*if(!mpz_cmp_ui(d, 1)){
			printf("Larger\n");
		}else*/ if(!mpz_cmp(d, num)){
			printf("Smaller\n");
		}

		// Result
		if(mpz_cmp_ui(d, 1) && mpz_cmp(d, num)){
			gmp_printf("Factor: %Zd\n", d);
			break;
		}
	}

	mpz_clears(M, a, d, 0);

	return 0;
}

/*	Pollard-brent algorithm, for given `num`
	Adapted from pascal code in "An improved monte carlo factorization algorithm" - R.P. Brent
	*Should* be working. It appears to be
*/
long brent(const mpz_t num, gmp_randstate_t state, const int c_val){
	mpz_t xn, y, q, x, G;
	mpz_t ys;
	mpz_t t1,t2; // Intermediate variables
	mpz_t c; // Constant for function

	uint64_t i;
	uint64_t r = 1;
	uint64_t k, m = 1; // m looks like an incrementor of sorts, not sure

	mpz_inits(xn, y, q, x, G, 0);
	mpz_init(ys);
	mpz_init(t1);
	mpz_init(t2);
	mpz_init(c);

	mpz_set_ui(xn, 2);
	mpz_set(y, xn);
	mpz_set_ui(q, 1); // q appears to be running product
	mpz_set_ui(c, c_val);

	do{
		// x = y
		mpz_set(x, y);

		for(i = 1; i <= r;++i){
			// y = f(y) ==> y = y^2 + 1 (mod N)
			mpz_mul(y, y, y); // t1 = y^2
			mpz_add(y, y, c); // t2 = t1 + c
			mpz_mod(y, y, num); // y = t2 (mod N)
		}
		k = 0;

		do{
			// ys = y
			mpz_set(ys, y);

			// y = f(y)
			mpz_mul(y, y, y);
			mpz_add(y, y, c);
			mpz_mod(y, y, num);

			// q = q * |x - y| (mod N)
			mpz_sub(t1, x, y);
			mpz_mul(t2, q, t1);
			mpz_mod(q, t2, num);

			k = k + 1;
		}while(k < r);

		// G = GCD(q, N)
		mpz_gcd(G, q, num);

		r = 2 * r;
	}while(mpz_cmp_ui(G, 1) == 0);

	if(mpz_cmp(G, num) == 0){
		do{
			// ys = f(ys)
			mpz_mul(ys, ys, ys);
			mpz_add(ys, ys, c);
			mpz_mod(ys, ys, num);

			// G = GCD(|x - ys|, N)
			mpz_sub(t1, x, ys);
			mpz_gcd(G, t1, num);
		}while(mpz_cmp_ui(G, 1) == 0);
	}
	if(mpz_cmp(G, num) == 0){
		printf("Failed\n");
	}else{
		gmp_printf("Factor: %Zu\n", G);
	}

	mpz_clears(xn, y, q, x, G, 0);
	mpz_clear(ys);
	mpz_clear(t1);
	mpz_clear(t2);
	mpz_clear(c);

	return 0;
}

/*	Get random data to seed generator
*/
long init_rand(gmp_randstate_t state){
	mpz_t seed;
	mpz_init(seed);

	const int SEED_COUNT = 16;
	const int SEED_SIZE = 4;
	char *buffer = malloc(SEED_COUNT * SEED_SIZE);
	if(buffer == NULL){
		perror("Can't malloc seed buffer");
		return -2;
	}

	// Use system entropy pool for seed
	printf("Fetching %lu bytes of random data..\n", SEED_COUNT*SEED_SIZE);
	size_t ret = getrandom(buffer, SEED_COUNT*SEED_SIZE, 0);
	if(ret != SEED_COUNT*SEED_SIZE){
		fprintf(stderr, "Random bytes read error\n");
		return -3;
	}

	// Convert random data to GMP integer, and initialize RNG with seed
	mpz_import(seed, SEED_COUNT, 1, SEED_SIZE, 0, 0, buffer);
	gmp_randinit_mt(state);
	gmp_randseed(state, seed);
	free(buffer);

	mpz_clear(seed);
}

int main(int argc, char *argv[]){
	char NUM_POS = -1;
	int c_val = 1;
	enum ops {OP_RHO, OP_P1, OP_BRENT} op;

	// Parse args
	for(int i = 1;i < argc;i++){
		if(argv[i][0] != '-'){
			NUM_POS = i;
			break; // Stop parsing for flags after first non-flag
		}

		char *tmp;
		long temp;
		switch(argv[i][1]){
			// p-1 B variable
			case 'b':
				if(++i >= argc){
					fprintf(stderr, "No max given.\n");
					return -2;
				}
				temp = strtol(argv[i], &tmp, 10);
				if(tmp != argv[i]) MAX_B = temp;
				break;
			// Brent constant flag
			case 'c':
				if(++i >= argc){
					fprintf(stderr, "No max given.\n");
					return -2;
				}
				temp = strtol(argv[i], &tmp, 10);
				if(tmp != argv[i]) c_val = temp;
				break;
			// Operation/method flags
			case 'R':
				op = OP_RHO;
				break;
			case 'P':
				op = OP_P1;
				break;
			case 'B':
				op = OP_BRENT;
				break;
			default:
				fprintf(stderr, "Unknown flag '%c'\n", argv[i][1]);
				return -2;
		}
	}

	if(NUM_POS < 0){
		fprintf(stderr, "No number given.\n");
		return -1;
	}

	// Initialization of GMP variables
	gmp_randstate_t state;
	mpz_t NUM;
	mpz_inits(NUM, 0);

	long res = mpz_set_str(NUM, argv[NUM_POS], 0);
	if(res){
		printf("Data conversion error.\n");
		return -1;
	}

	init_rand(state);

	gmp_printf("Num: %Zd\n", NUM);

	if(op == OP_RHO){
		printf("Using Pollard-Rho.\n");
		rho(NUM, state);
	}else if(op == OP_P1){
		printf("Using Pollard p-1.\n");
		gmp_printf("MAX_B: %d\n", MAX_B);
		printf("Prime gen..\n");
		res = gen_primes(MAX_B, &primes);
		printf("%lu primes generated\n", res);

		p1(NUM);
	}else if(op == OP_BRENT){
		printf("Using Pollard-Brent. c = %d\n", c_val);
		brent(NUM, state, c_val);
	}else{
		fprintf(stderr, "No op specified. Use -R, -P, or -B\n");
	}

	// Clean up
	free(primes);
	gmp_randclear(state);
	mpz_clears(NUM, 0);

	return 0;
}
