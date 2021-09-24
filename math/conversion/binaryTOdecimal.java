import java.util.Scanner;
public class binaryTOdecimal{
	
	public static void main(String[] args){
		
		int a=2,t=0;
		Scanner s = new Scanner(System.in);
		System.out.printf("Enter binary number to convert: ");
		String b = s.next();
		
		a=(int)Math.pow(a,(b.length()-1));
		for(char c:b.toCharArray()){
			if(c=='1'){
				t+=a;
				a/=2;
			}else{
				a/=2;
				continue;
			}
		}
		System.out.printf("%d\n",t);
	}
	
}
