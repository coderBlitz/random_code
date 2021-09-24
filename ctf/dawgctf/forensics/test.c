#include<stdint.h>
#include<stdio.h>

int main(){
	FILE *fp = fopen("ascii.txt", "r");

	uint64_t c = 0;
	uint8_t extra = 0;
	uint8_t count = 0;
	char buffer[64];
	uint8_t block = 0xFF;
	uint64_t test = 0;
	while((c = fgetc(fp)) != EOF){
		test |= (c & 0xFF) << (8*(count % 8));

		printf("test: %lx (%x)\n", test, c);

		if(extra & count % 8 == 0){
			for(int i = 0;i < 8;++i){
				c = (test >> i) & 0xFF;
				buffer[count+i-8] = c;
			}
			printf("%s\n", buffer);
			test = 0;
		}
		extra = 1;
		++count;
	}
	printf("%s\n", buffer);

	fclose(fp);
}
