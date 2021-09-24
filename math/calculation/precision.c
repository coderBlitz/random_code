#include<stdio.h>
int main(){
  double a = 9.4 - 9.0 - 0.4;
  double b = 9.4;
  double c = 0.4;
  double d = 9.0;

  long *ap = (long *)&a;

  /*scanf("%lf", &b);
  printf("%lf\n", b);
  scanf("%lf", &c);
  scanf("%lf", &d);*/

  printf("a = %lf = 0x%lX\n", a, *ap);
  printf("b-c-d=%e\n", b-c-d);
  printf("a = %e\n", a);
  printf("b-d-c=%e\n", b-d-c);
}
