import java.util.Scanner;

/* Simple Summation program
   Summation is: startNum+(startNum+1)+(startNum+2)...(end-2)+(end-1)+end
   Example: summation of 1 to 5 is 15
   1+2+3+4+5 = 15
*/

public class summation{
	public static void main(String args[]){
		int start,end;
		Scanner s = new Scanner(System.in);

		System.out.printf("Enter the number to start at: ");
		start = s.nextInt();
		System.out.printf("Enter the number to end at: ");
		end = s.nextInt();

		int n = (end-start)+1;

		long sum = (long)((n/2.0)*(start+end));

		System.out.printf("%d\n",sum);
	}
}
