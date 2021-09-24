#include<stdio.h>
#include<stdlib.h>

int main(){
  unsigned long long N = 0;
  unsigned long long a=0,b=1,c=0;
  unsigned long long res = 0;

  printf("Enter number to find fibonacci of: ");
  scanf("%llu",&N);

  for(long long i=1;i<N;i++){
    c = a+b;
    a = b;
    b = c;
  }
  printf("Fib(%llu) = %llu\n",N,c);
}
