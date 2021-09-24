import java.util.Scanner;

public class decToHex{
	public static void main(String args[]){
		Scanner s = new Scanner(System.in);
		System.out.printf("Enter integer to convert to hex: ");
		int a = s.nextInt();
		System.out.printf("%x\n",a);
	}
}
