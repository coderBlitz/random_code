#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

// Max out memory
// Although system may kill program

int main(){
	int *i;
	while(1){
		i = malloc(sizeof(int));
		if(i == NULL) break;
	}
	sleep(60);
}
