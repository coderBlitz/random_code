#include<stdio.h>
#include<stdlib.h>

void multiBy2(int i){// Function to be used by secondMain()
	printf("%d*2=%d\n",i,i*2);
}
void add5(int i){// Function to be used by secondMain()
	printf("%d+5=%d\n",i,i+5);
}

void secondMain(void (*func)(int)){// Argument is void function pointer that takes 1 int arg
	int n=6;// Will pass this number to function given
	func(n);// Call function pointer given as parameter
}


// Function pointer format:
// functionReturnType (*ptrName)(arg1Type,arg2Type);
int main(){
	void (*add)(int);// void function pointer 1
	void (*multi)(int);// void function pointer 2

	add = &add5;// Point to add2()
	multi = &multiBy2;// Point to multiplyBy5()

	// Change to 'add'/'add5' or 'multi'/'multiBy2' to change which method is called
	secondMain(multiBy2);

return 0;
}
