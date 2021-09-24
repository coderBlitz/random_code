#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main(){
	printf("Standard Deviation\n");
	int numOfValues=0;
	float variance,total;

	printf("How many values total: ");
	scanf("%d",&numOfValues);

	float num[numOfValues];

	for(int i=0;i<numOfValues;i++){
		printf("Enter value #%d: ",i+1);
		float tmp;
		scanf("%f",&tmp);
		num[i] = tmp;
		total += tmp;
	}
	float mean,stdDeviation;
	mean = total/numOfValues;

	for(int n=0;n<numOfValues;n++){
		variance += pow((num[n]-mean),2);
	}
	variance /= numOfValues;//Full EQ: sumOfDifferencesBetweenValuesAndMeanSquared/numOfValues
	stdDeviation = sqrt(variance);//Square root of this /\/\

	printf("%d Values entered\nMean: %.3f\n",numOfValues,mean);
	printf("Variance: %.3f\nStandard Deviation: %.3f\n",variance,stdDeviation);
	printf("Standard Deviation range: %.3f—%.3f\n",mean-stdDeviation,mean+stdDeviation);
}
