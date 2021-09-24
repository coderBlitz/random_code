#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

char *set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int main(){
	char *str = "Man is distinguished, not only by his reason, but by this singular passion from other animals, which is a lust of the mind, that by a perseverance of delight in the continued and indefatigable generation of knowledge, exceeds the short vehemence of any carnal pleasure.";
	uint32_t len = strlen(str);
	uint8_t mod = len % 3;
	printf("Length: %u (%u)\n", len, mod);

	uint32_t store = 0;

	for(uint32_t offset = 0;offset < len;offset += 3){
		store = 0;

		for(uint8_t i = 0;i < 3;++i){
			if((offset + i) < len) store = store | str[offset + i];
			store = store << 8;
		}

//		printf("Store: %u\n", store);
		uint32_t temp = 0;
		char c = 0;
		char out[4] = {0,0,0,0};
		for(uint8_t i = 0;i < 4;++i){
			temp = store;
			temp = temp << (6*i);
			temp = temp >> 26;
			if(temp == 0){
				if((mod == 1 && i > 1) || (mod == 2 && i == 3)) c = '=';
				else c = set[0];
			}else c = set[temp];
//			printf("Temp %u: %u (%c)\n", i, temp, c);
			out[i] = c;
		}
		printf("%s", out);
	}
	printf("\n");
}
