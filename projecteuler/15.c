#include<stdio.h>
#include<stdint.h>

uint64_t lat(uint64_t n, uint64_t m){
	if(n == 0 || m == 0) return 0;
	if(n == 1 || m == 1) return m + n;

	return lat(n-1, m) + lat(n, m-1);
}

int main(int argc, char *argv[]){
	int N = 20;

	for(int i = 1;i <= N;++i){
		printf("%d ==> %lu\n", i, lat(i, i));
	}
}
