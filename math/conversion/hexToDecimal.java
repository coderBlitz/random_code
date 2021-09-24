import java.util.Scanner;

public class hexToDecimal{
	public static void main(String args[]){
		int a; Scanner s = new Scanner(System.in);
		System.out.printf("Enter the Hex number to convert: ");
		a = s.nextInt(16);// 16 is radix or base in practical terms
		System.out.printf("%d\n",a);
	}
}
