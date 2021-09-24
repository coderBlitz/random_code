#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<openssl/sha.h>
#include"base64.h"

char *set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int main(){
	printf("Length: %d\n", SHA512_DIGEST_LENGTH);

	char data[] = "abcd";
	unsigned char hash[SHA512_DIGEST_LENGTH];

	SHA512(data, sizeof(data) - 1, hash);

//	int len = 0;
//	char *result = base64((void *)hash, SHA512_DIGEST_LENGTH, &len);
//	printf("len: %d\n%s\n", len,result);

	char *target = "18e4aaa17638e9f8e0dd7cbf0b975ae25aac97163c7b1d3d44c205e5cd80dd8a92377cd26d439601c4e7d1aaf46a218067fa528d3bf96fb46f5e932b43c30099";

	unsigned char hash_asc[2*SHA512_DIGEST_LENGTH];
	hash_asc[0] = 0;
	for(int i=0;i < SHA512_DIGEST_LENGTH;i++){
		sprintf(hash_asc + strlen(hash_asc), "%02x", hash[i]);
	}
	printf("\n");

	printf("%.*s\n",128, hash_asc);

	if(strncmp(hash_asc, target, SHA512_DIGEST_LENGTH*2) == 0){
		printf("Found match: '%.*s'\n",sizeof(data)-1,data);
		printf("Target was '%s'\n",target);
		exit(0);
	}

//	printf("%s\n",target);
}
