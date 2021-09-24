#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<pthread.h>
#include<unistd.h>

/* 32 - Space
   48 - '0'
   57 - '9'
   65 - 'A'
   90 - 'Z'
   97 - 'a'
   122 - 'z'
   126 - '~'
*/

static const int min = 48;
static const int max = 126;
static const int MAX_LENGTH = 7;// Length of the strings to generate

const char *target = "a9f51566bd6705f7ea6ad54bb9deb449f795582d6529a0e22207b8981233ec58";

void test(char i){
	i %= 255;
//	printf("Hello world from %c\n",i);


	unsigned char pw[MAX_LENGTH];
//	int len = MAX_LENGTH;// If trying all lengths up to MAX_LENGTH, just add for loop
	for(int len = 1;len <= MAX_LENGTH;len++){
		printf(" %d Starting length %d\n",i ,len);
		pw[0] = i + min;
		for(int i = 1;i < len;i++) pw[i] = min;
//		printf("%d pw: %.*s\n",i,len,pw);// The dot means length, the star means take argument

		while(1){// Loop until broken out of
			if(strncmp(pw, target, len) == 0){
				printf("Found match: %.*s\n",len,pw);
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
			if(pw[0] > i+min) break;// If the first character gets incremented, it's done
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
