#include<stdio.h>
#include<string.h>
#include"threads.h"

void hi(int tid, void *args){
	char *name = (char *)args;

	printf("Hello %s from %ld!\n", name, tid);
}

void num(int tid, void *args){
	long num = (long) args;

	printf("%5ld from %3d\n", num, tid);
}

int main(int argc, char *argv[]){
	struct thread_pool tp;
	printf("Initializing..\n");
	int nt = 0;
	if(argc >= 2){
		nt = strtoul(argv[1], NULL, 10);
	}
	int np = pool_init(&tp, nt);
	printf("%2d threads available\n", np);

	char *m1 = "main";
	pool_push(&tp, hi, m1);
	char *m2 = "number 2";
	pool_push(&tp, hi, m2);
	char *m3 = "number 3";
	pool_push(&tp, hi, m3);

	for(int i = 0;i < 50;i++){
		pool_push(&tp, num, (void *)(long)i);
	}
	pool_clear(&tp);
	for(int i = 0;i < 50;i++){
		pool_push(&tp, num, (void *)(long)i);
	}

	pool_start(&tp);

	pool_finish(&tp, 0);

	pool_stop(&tp, 0);

	pool_destroy(&tp);

	return 0;
}
