#include<stdio.h>
#include<stdlib.h>
#include<signal.h>
#include<setjmp.h>

volatile int a;
jmp_buf jbf;
void trap_handle(int dummy){
	printf("Dummy: %d\t%d\t%d %d %d %d\n", dummy, FPE_INTDIV, FPE_INTOVF, FPE_FLTDIV, FPE_FLTSUB, FPE_FLTINV);
	a = 2;
	//longjmp(jbf, 2); // Only use if setjmp is called
}

int main(int argc, char *argv[]){
	if(argc < 2) return 1;

	signal(SIGFPE, trap_handle);

	printf("To infinity and beyond..\n");

	a = strtod(argv[1], NULL);
	int b = 8;
	printf("Before\n");

	//setjmp(jbf); // Allows graceful completion

	int c = b/a;
	printf("c = %d\n", c);
}
