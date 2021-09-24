#include<stdio.h>

int factorial(int i){
	for(int j=i-1;j>0;j--){
		i *= j;
	}
return i;
}

int main(){
	int total,pieces,possibilities;
	printf("Number of Possibilities\nHow many objects are there: ");
	scanf("%d",&pieces);
	printf("How many total spaces are there including ones with pieces on them: ");
	scanf("%d",&total);

	possibilities = factorial(total)/(factorial(total-3)*factorial(pieces));
	printf("There are %d unique possibilities\n",possibilities);
}
