/* Not currently in the works

*/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

char* addLargeNumbers(char *a,char *b){
	char *result = malloc(strlen(a)+strlen(b)+1);
	int s=0;
	for(int n=strlen(a)-1; n>=0; n--){//Method trys to add numbers in a string, to deal with incredibly
		int t=0;		//large numbers
		int i = a[i] - '0';
		int j = b[i] - '0';
		t= (i*j)+s;
		s=0;
		while(t>10){
			t -= 10;
			s += 1;
		}
	}
	return result;
}

int main(){
	char *a = "120";
	char *b = "23";
	printf("%s\n", addLargeNumbers(a,b));
}
