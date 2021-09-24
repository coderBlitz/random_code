import java.util.Scanner;

/* Find the standard deviation of given values
   Converted from C to Java. 
*/
public class deviation{
	public static void main(String args[]){
		System.out.printf("Standard Deviation\n");
		int numOfValues=0;
		double variance=0.0,total=0.0;

		Scanner s = new Scanner(System.in);

		System.out.printf("How many values total: ");
		numOfValues = s.nextInt();

		double num[] = new double[numOfValues];

		for(int i=0;i<numOfValues;i++){
			System.out.printf("Enter value #%d: ",i+1);
			double tmp;
			tmp = s.nextDouble();
			num[i] = tmp;
			total += tmp;
		}

		double mean,stdDeviation;
		mean = total/numOfValues;

		for(int n=0;n<numOfValues;n++){
			variance += Math.pow((num[n]-mean),2);
		}
		variance /= numOfValues;//Full EQ: sumOfDifferencesBetweenValuesAndMeanSquared/numOfValues
		stdDeviation = Math.sqrt(variance);//Square root of this /\/\

		System.out.printf("%d Values entered\nMean: %.3f\n",numOfValues,mean);
		System.out.printf("Variance: %.3f\nStandard Deviation: %.3f\n",variance,stdDeviation);
		System.out.printf("Standard Deviation range: %.3f—%.3f\n",mean-stdDeviation,mean+stdDeviation);
	}
}
