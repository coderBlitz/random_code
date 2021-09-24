#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<pthread.h>
#include<unistd.h>
#include<openssl/sha.h>

/* 32 - Space
   48 - '0'
   57 - '9'
   65 - 'A'
   90 - 'Z'
   97 - 'a'
   122 - 'z'
   126 - '~'
*/

static const int min = 97;
static const int max = 122;
static const int MAX_LENGTH = 6;// Length of the strings to generate

const char *target = "84f42808be927a3c9012bbd2613520a801866998fba432849274f1916c002db90ae060f450333a4c43b1e71990ff72a1339fe2b2825ce827ad46a778344582b4";
//const char *target = "d8022f2060ad6efd297ab73dcc5355c9b214054b0d1776a136a669d26a7d3b14f73aa0d0ebff19ee333368f0164b6419a96da49e3e481753e7e96b716bdccb6f";

void print_hash(const char hash[], char hash_asc[]){
	for(int i=0;i < SHA512_DIGEST_LENGTH;i++){
		sprintf(hash_asc + strlen(hash_asc), "%02x", hash[i] & 0xFF);
	}
}

void test(char id){
	id %= 255;
//	printf("Hello world from %c\n",i);


	unsigned char pw[MAX_LENGTH+1];
//	int len = MAX_LENGTH;// If trying all lengths up to MAX_LENGTH, just add for loop
	for(int len = MAX_LENGTH;len <= MAX_LENGTH;len++){
		printf(" %d Starting length %d\n",id ,len);
		pw[0] = id + min;
		pw[MAX_LENGTH+1] = 0;// Null terminator
		for(int i = 1;i < len;i++) pw[i] = min;

//		printf("%d pw: %s\n",i,pw);// The dot means length, the star means take argument

		unsigned char hash[SHA512_DIGEST_LENGTH];
		while(1){// Loop until broken out of
			unsigned char hash_asc[SHA512_DIGEST_LENGTH*2];
			hash_asc[0] = 0;

			SHA512(pw, len, hash);// Get hash

			print_hash(hash, hash_asc);
//			printf("'%s' - %2d: %.*s\n",pw, strncmp(hash_asc, target, SHA512_DIGEST_LENGTH*2),128, hash_asc);

			if(strncmp(hash_asc, target, SHA512_DIGEST_LENGTH*2) == 0){
				printf("Found match: '%.*s'\n",len, pw);
				printf("Target was '%s'\n",target);
				exit(0);
			}

			for(int i = len-1;i >= 0;i--){
				pw[i]++;
				if(pw[i] > max){
					pw[i] = min;
					continue;
				}else break;
			}
			if(pw[0] > id+min) break;// If the first character gets incremented, it's done
		}
//		printf("%d done! Pw: %.*s\n",i,len,pw);
	}
}

int main(){
	int N = max-min;
	pthread_t pointers[N];

	void (*func) = test;// Function pointer to test

	printf("Attempting to spawn %d threads to generate string length %d..\n",N,MAX_LENGTH);
	for(int i = 0;i < N;i++){
//		printf("Creating thread %d\n",i);
		pthread_create(&pointers[i],NULL,func,(void *)(long)i);
//		printf("Created!\n");
	}

	printf("Waiting for threads to finish..\n");
	for(int i = 0;i < N;i++){
		pthread_join(pointers[i], NULL);
	}
	printf("Done\n");
}
