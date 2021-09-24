/***	rsa.c -- Simple RSA implementation using custom extended euclidean and miller-rabin
	Author: Chris Skane
	Date: 22 APR 2020
	Build: Use included Makefile, or simply run `gcc -o rsa.out rsa.c -lgmp`
	Notes: Uses same algorithm for all bit sizes and round count

	Command-line options are as follows:
		Encryption/decryption (currently mutually exclusive from kegen)
			'-E' -- Encrypt, expects at least the modulus to be given. Encrypts last parameter
			'-D' -- Decrypt, expects at least the modulus to be given. Decrypts last parameter
			'-N' -- Modulus, sets the modulus for encryption/decryption
			'-e' -- Exponent, sets the exponent for encryption/decryption
			'-d' -- Exponent, equivalent to '-e'
		Key generation
			'-m' -- Bits, the size in bits of the modulus N to generate
			'-s' -- Rounds, sets how many rounds each miller-rabin test should use
***/

#include<errno.h>
#include<gmp.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/random.h>
#include<time.h>
#include<unistd.h>

long RSA_SIZE = 1024;
long ROUNDS = 200; // Rounds used in miller rabin


/**	GCD using extended euclids algorithm. Iterative.
	res === (A,B)
	res = A*X + B*Y		(A > B)
	res = B*X + A*Y		(A < B)
**/
uint64_t GCD(mpz_t res, mpz_t A, mpz_t B, mpz_t X, mpz_t Y){
	// Trivial cases of both 0, or either 0
	if(mpz_cmp_ui(A, 0) == 0 && mpz_cmp_ui(B, 0) == 0){
		mpz_set_ui(res, 0);
		return 0;
	}else if(mpz_cmp_ui(A, 0) == 0 || mpz_cmp_ui(B, 0) == 0){
		if(mpz_cmp_ui(A, 0)) mpz_set(res, A);
		else mpz_set(res, B);
		return 0;
	}

	mpz_t remainder, quotient;
	mpz_t C, D;
	mpz_inits(remainder, quotient, C, D, 0);

	// Always have D > C to start
	if(mpz_cmp(A, B) > 0){
		mpz_set(D, A);
		mpz_set(C, B);
	}else{
		mpz_set(D, B);
		mpz_set(C, A);
	}

	char flip = 0; // Which row is being modified in scalars matrix
	char val;

	mpz_t scalars[4]; // Scalars representing the matrix form of A,B
	mpz_inits(scalars[0], scalars[1], scalars[2], scalars[3], 0);

	mpz_set_ui(scalars[0], 1);
	mpz_set_ui(scalars[1], 0);
	mpz_set_ui(scalars[2], 0);
	mpz_set_ui(scalars[3], 1);

	// Remainder loop
	do{
		mpz_fdiv_qr(quotient, remainder, D, C);
		//gmp_printf("%Zu = %Zu(%Zu) + %Zu\n", D, C, quotient, remainder);

		val = 2*flip; // Saves 4 multiplications. Used for indexing
		mpz_submul(scalars[val], quotient, scalars[2 - val]); // s[0/2] -= q * s[2/0]
		mpz_submul(scalars[val + 1], quotient, scalars[3 - val]); // s[1/3] -= q * s[3/1]

		mpz_swap(D,C);
		mpz_set(C, remainder);
		flip = 1 - flip; // Flip to opposite row
	}while(mpz_cmp_d(remainder, 0));

	//gmp_printf("%Zd * %Zu + %Zd * %Zu = 1\n", scalars[0], A, scalars[1], B);
	//gmp_printf("%Zd * %Zu + %Zd * %Zu = %Zd\n", scalars[2], A, scalars[3], B, C);

	// Set results depending on which row was the last operation
	if(mpz_cmp_ui(D, 1) == 0 && flip == 0){
		//gmp_printf("%Zd * %Zu + %Zd * %Zu = 1\n", scalars[0], A, scalars[1], B);
		mpz_set(X, scalars[0]);
		mpz_set(Y, scalars[1]);
	}else{
		//gmp_printf("%Zd * %Zu + %Zd * %Zu = %Zd\n", scalars[2], A, scalars[3], B, C);
		mpz_set(X, scalars[2]);
		mpz_set(Y, scalars[3]);
	}

	mpz_set(res, D); // Set result

	// Cleanup
	mpz_clears(remainder, quotient, C, D, 0);
	mpz_clears(scalars[0], scalars[1], scalars[2], scalars[3], 0);

	return 0;
}

// Returns 0 for likely prime, returns 1 for composite
uint64_t miller_rabin(const mpz_t N, uint64_t s, gmp_randstate_t state){
	if(mpz_even_p(N)) return 1; // Even numbers automatically non-prime.
	if(mpz_cmpabs_ui(N, 3) <= 0) return 0; // These are automtically prime
	

	mpz_t A, B, C, R, V;
	mpz_inits(A, B, C, R, V, 0);

	mpz_sub_ui(V, N, 1);
	//gmp_printf("MR testing: %Zu\n", V);

	// Repeat test s times
	for(uint64_t i = 0;i < s;++i){
		mpz_urandomm(A, state, V);
		//gmp_printf("%Zd\n", A);

		/* "Witness" computation */
		// Compute r,t
		mp_bitcnt_t t = mpz_scan1(V, 0);
		mpz_fdiv_q_2exp(R, V, t);

		// Compute b
		mpz_powm(B, A, R, N); // B = a^r mod N
		mpz_set(C, B); // Initialize C in case t = 0

		for(uint64_t j = 1;j <= t;++j){
			mpz_powm_ui(C, B, 2, N); // C = B^2 mod N

			// If C == 1 and B != +-1 (mod N)
			if(mpz_cmp_ui(C, 1) == 0 && mpz_cmp_ui(B, 1) && mpz_cmp(B, V)){
				//printf("Composite\n");
				return 1;
			}
			mpz_set(B, C);
		}
		if(mpz_cmp_ui(C, 1)) return 1; // Composite
		/* End "witness" */
	}

	mpz_clears(A, B, C, R, V, 0);
	return 0;
}

void genRSA(gmp_randstate_t state){
	/*
		1) Generate 2 random primes P,Q of RSA_SIZE bits
		2) Calculate N = PQ
		3) Calculate Phi(N) = (P-1)(Q-1)
		4) Choose exponent e co-prime to Phi(N). Likely use common 65537 (0x010001)
		5) Compute d, the multiplicative inverse of exponent modulo Phi(N)
		*) Public key is (e, N)
		*) Private key is (d, N)
	*/

	mpz_t P, Q, e, N, d, phi;
	mpz_inits(P, Q, e, N, d, phi, 0);

	// Gen random number and test primality. Then increment till success
	/*mpz_urandomb(P, state, RSA_SIZE/2);
	if(mpz_even_p(P))
		mpz_add_ui(P, P, 1); // Even automatically composite, so go to next odd
	while(miller_rabin(P, ROUNDS, state)){
		mpz_add_ui(P, P, 2);
	}*/

	// Safe prime method
	do{
		mpz_urandomb(P, state, RSA_SIZE/2 - 1);
		mpz_mul_ui(P, P, 2);
		mpz_add_ui(P, P, 1);
	}while(miller_rabin(P, ROUNDS, state));

	// Generate Q
	/*mpz_urandomb(Q, state, RSA_SIZE/2);
	if(mpz_even_p(Q))
		mpz_add_ui(Q, Q, 1);
	while(miller_rabin(Q, ROUNDS, state)){
		mpz_add_ui(Q, Q, 2);
	}*/
	do{
		mpz_urandomb(Q, state, RSA_SIZE/2 - 1);
		mpz_mul_ui(Q, Q, 2);
		mpz_add_ui(Q, Q, 1);
	}while(miller_rabin(Q, ROUNDS, state));

	mpz_mul(N, P, Q); // N = PQ

	mpz_sub_ui(P, P, 1); // P-1
	mpz_sub_ui(Q, Q, 1); // Q-1

	mpz_mul(phi, P, Q); // Phi(N) = (P-1)(Q-1)

	mpz_set_ui(e, 65537); // Use common exponent instead of finding one

	// Since Phi(N) is almost certainly greater than e, this is order of parameters
	GCD(P, phi, e, Q, d); // P,Q not needed to be original values anymore

	// While testing, use library code to verify
	//printf("Verifying..\n");
	//mpz_invert(P, e, phi); // Use library function to test results
	//gmp_printf("P: %Zd\n", P);
	
	// If inverse is negative, add phi to make it positive.
	if(mpz_cmp_d(d, 0) < 0){
		mpz_add(d, d, phi);
	}
	gmp_printf("N: %Zu\ne: %Zd\nd: %Zd\n", N, e, d);

	mpz_clears(P, Q, e, N, d, phi, 0);
}

// Converts message to number (direct, not interpreted), then encrypts
long rsa_encrypt(const mpz_t exponent, const mpz_t modulus, const void *message, const size_t count, const size_t size){
	mpz_t data, result;
	mpz_inits(data, result, 0);

	//printf("Data count: %lu\tsize: %lu\n", count, size);
	mpz_import(data, count, 1, size, 0, 0, message);

	gmp_printf("data: 0x%Zx\n", data);

	mpz_powm(result, data, exponent, modulus);

	gmp_printf("result: 0x%Zx\n", result);

	mpz_clears(data, result, 0);
}

// Interprets given message as number, decimal octal or hex, then decrypts
void * rsa_decrypt(const mpz_t exponent, const mpz_t modulus, const void *message, size_t *size){
	mpz_t data, result;
	mpz_inits(data, result, 0);

	long res = mpz_set_str(data, message, 0);
	if(res){
		printf("Data conversion error.\n");
		return NULL;
	}

	//gmp_printf("data: 0x%Zx\n", data);

	mpz_powm(result, data, exponent, modulus);

	//gmp_printf("result: 0x%Zx\n", result);

	void *ret = mpz_export(NULL, size, 1, 1, 0, 0, result); // (rop, countp, order, size, endian, nails, op)

	mpz_clears(data, result, 0);

	return ret;
}

int main(int argc, char *argv[]){
	// Initialization of GMP variables
	gmp_randstate_t state;
	mpz_t Az, Bz, Cz, rseed, N, e;
	mpz_inits(Az, Bz, Cz, rseed, 0);
	mpz_inits(N, e, 0);

	mpz_set_ui(e, 65537); // Have default value

	char ENCRYPT_MSG = 0;
	char DECRYPT_MSG = 0;
	char GEN_KEY = 1;
	char MODULUS = 0;
	char EXPONENT = 0;
	int MSG_IDX = -1;

	// Parse cmdline flags
	for(int i = 1;i < argc;i++){
		if(argv[i][0] != '-'){
			MSG_IDX = i;
			break; // Ignore non-flags
		}

		char *tmp;
		long temp;
		switch(argv[i][1]){
			// Modulus bit size flag
			case 'm':
				if(++i >= argc){
					fprintf(stderr, "No size given.\n");
					return -2;
				}
				temp = strtol(argv[i], &tmp, 10);
				if(tmp != argv[i]) RSA_SIZE = temp;
				break;
			// Num rounds flag
			case 's':
				if(++i >= argc){
					fprintf(stderr, "No rounds given.\n");
					return -2;
				}
				temp = strtol(argv[i], &tmp, 10);
				if(tmp != argv[i]) ROUNDS = temp;
				break;
			// Modulus flag
			case 'N':
				if(++i >= argc){
					fprintf(stderr, "No modulus given.\n");
					return -2;
				}
				if(!mpz_set_str(N, argv[i], 0)) MODULUS = 1;
				break;
			// Exponent flag(s)
			case 'd':
			case 'e':
				if(++i >= argc){
					fprintf(stderr, "No exponent given.\n");
					return -2;
				}
				if(!mpz_set_str(e, argv[i], 0)) EXPONENT = 1;
				break;
			// Operation cases
			case 'D':
				DECRYPT_MSG = 1;
				break;
			case 'E':
				ENCRYPT_MSG = 1;
				break;
			default:
				fprintf(stderr, "Unknown flag '%c'\n", argv[i][1]);
		}
	}

	if(ENCRYPT_MSG || DECRYPT_MSG || MODULUS || EXPONENT) GEN_KEY = 0;

	if(!GEN_KEY && !(ENCRYPT_MSG || DECRYPT_MSG)){
		fprintf(stderr, "Neither operation fully determined. Please give all necessary parameters.\n");
		return -2;
	}else if(!GEN_KEY && MSG_IDX < 0){
		fprintf(stderr, "No message given.\n");
		return -2;
	}

	if(GEN_KEY){
		// Buffer to store random seed data
		const int SEED_COUNT = 32;
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
		mpz_import(rseed, SEED_COUNT, 1, SEED_SIZE, 0, 0, buffer);
		gmp_randinit_default(state);
		gmp_randseed(state, rseed);
		free(buffer);

		printf("Modulus size: %d\n", RSA_SIZE);
		printf("Rounds: %d\n", ROUNDS);

		printf("Generating values..\n");
		genRSA(state); // TODO: Make genRSA return key and exponent, to generate AND encrypt in one call (maybe)

		gmp_randclear(state);
	}else if(ENCRYPT_MSG){
		if(!MODULUS){
			fprintf(stderr, "Must provide a modulus through '-N'\n");
			return -2;
		}
		printf("Encrypting message..\n");

		rsa_encrypt(e, N, argv[MSG_IDX], strlen(argv[MSG_IDX]), 1);
	}else if(DECRYPT_MSG){
		printf("Decrypting message..\n");
		if(!EXPONENT){
			fprintf(stderr, "Must provide an exponent through '-d'\n");
			return -2;
		}

		size_t len = 0;
		char *msg = rsa_decrypt(e, N, argv[MSG_IDX], &len);
		msg[len] = 0;

		if(msg != NULL) printf("Message: '%s'\n", msg);
		free(msg);
	}


	// Clean up
	mpz_clears(Az, Bz, Cz, rseed, 0);
	mpz_clears(N, e, 0); // Split because segfault with all in one

	return 0;
}
