#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define Ch(x, y, z) ((x & y)^(~x & z))
#define Maj(x, y, x) ((x & y) ^ (x & z) ^ (y & z))

uint64_t be64(uint64_t num){
	uint64_t value = 0;
	value |= (num << 56) & 0xFF00000000000000;
	value |= (num << 40) & 0x00FF000000000000;
	value |= (num << 24) & 0x0000FF0000000000;
	value |= (num << 8) &  0x000000FF00000000;
	value |= (num >> 8) &  0x00000000FF000000;
	value |= (num >> 24) & 0x0000000000FF0000;
	value |= (num >> 40) & 0x000000000000FF00;
	value |= (num >> 56) & 0x00000000000000FF;

	return value;
}

int main(int argc, char *argv[]){
	char *message = "abc";

	uint64_t message_length = strlen(message);
	uint64_t message_bits = message_length << 3;

	printf("Message length: %u (%u)\n", message_length, message_bits);

	uint8_t block[128];

	strncpy(block, message, message_length);

	uint16_t fill_length = 112 - message_length;
	for(uint64_t i = 1;i < fill_length;i++) block[message_length + i] = 0;

	block[message_length] = 0x80;
	uint64_t *thing = (uint64_t *)block;
	thing[15] = be64(message_bits);

	for(uint64_t i = 0;i < 128;i++) printf("%x", block[i]);
	printf("\n");

	uint64_t M[16];
	for(uint8_t i = 0;i < 128;i += 8){
		M[i/8] = be64(*((uint64_t *) &block[i]));
	}
	printf("Test: %lx\n", M[0]);

	uint64_t H[8] = {0x6a09e667f3bcc908,
					 0xbb67ae8584caa73b,
					 0x3c6ef372fe94f82b,
					 0xa54ff53a5f1d36f1,
					 0x510e527fade682d1,
					 0x9b05688c2b3e6c1f,
					 0x1f83d9abfb41bd6b,
					 0x5be0cd19137e2179};

	uint64_t A = H[0];
	uint64_t B = H[1];
	uint64_t C = H[2];
	uint64_t D = H[3];
	uint64_t E = H[4];
	uint64_t F = H[5];
	uint64_t G = H[6];
	uint64_t H = H[7];

	// Need to finish

	return 0;
}
