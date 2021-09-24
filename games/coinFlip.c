#include <stdio.h>
#include <stdlib.h>
#include<string.h>
#include<time.h>

int main(){
  
  int a=0,b=0;
  char c[7]="";
  
  printf("Heads or Tails?");
  scanf("%s",c);
  
  srand(time(0));
  b= (rand() % 2);
  
  if(strcmp(c,"heads") == 0){
    a=1;
    if(a==b){printf("Correct\n");}
    else {printf("Incorrect\n");}
  }
    else {a=0;printf("Incorrect\n");}
  
  while(strcmp("done",c) != 0) {    
    printf("Type \"done\" to quit.\n\n");
    printf("Heads or Tails?");
    scanf("%s",c);
    
    b= (rand() % 2);
    
    if(c=="Heads"||"heads"||"head"||"heads"){
	a=1;
	if(a==b){printf("Correct\n");}
	else {printf("Incorrect\n");}
    }
    if (strcmp(c,"done")) {exit(0);}
    else {a=0;printf("Incorrect\n");}
    
  }
  
  return 0;
}
