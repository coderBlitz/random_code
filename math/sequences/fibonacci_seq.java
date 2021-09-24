import java.util.Scanner;

public class fibonacci_seq{
	public static void main(String args[]){
		int tmp; Scanner scan = new Scanner(System.in);
		System.out.printf("Find number at N in fibonacci series: ");
		tmp = scan.nextInt();

		if(tmp == 1 || tmp == 2){
			System.out.printf("1\n");
			System.exit(0);
		}

		long first=1,second=1,num=0;
		for(int i=3;i<=tmp;i++){
			num = first + second;
			first = second;
			second = num;
		}
		System.out.printf("%d\n",num);
	}
}
