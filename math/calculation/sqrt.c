 #include<stdio.h>

// Calculates the square root of a number
// Using the babylonian method
double sqrt(double num){
//	printf("Num=%lf\n",num);
	int iterations=40;
	double result=1.0;
	for(int i=0;i<iterations;i++){
		result = ((num/result) + result)/2;
//printf("#%d: %.16lf\n",i,result);
	}
	return result;
}

int main(){
	double a;

	printf("Square Root Finder v1.0\nEnter Number: ");
	scanf("%lf",&a);

	printf("Square root of %lf is %.10lf\n",a,sqrt(a));
}
