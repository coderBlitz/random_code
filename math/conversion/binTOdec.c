#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//strrev() copied from stackoverflow
char *strrev(char *str){
      char *p1, *p2;

      if (! str || ! *str)
            return str;
      for (p1 = str, p2 = str + strlen(str) - 1; p2 > p1; ++p1, --p2)
      {
            *p1 ^= *p2;
            *p2 ^= *p1;
            *p1 ^= *p2;
      }
      return str;
}

int main(){
	
	int a=1,t=0;
	char b[17];
	
	printf("Enter binary to convert to decimal: ");
	scanf("%s",b);
	strrev(b);

	a=a<<(sizeof(b)-1);

	for(int i=(sizeof(b)-1); i>=0; i--){
		char c = b[i];
		//printf("C = %c\nA = %d\n",c,a);
		if(c=='1'){
			t+=a;
			//printf("adding\n");
			a/=2;
		}else{
			a/=2;
			continue;
		}
	}
	printf("%d\n",t);
}
