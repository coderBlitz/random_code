public class e{
	public static void main(String args[]){
		double a;
		int n=1;

		while(n<1e9){
			a = Math.pow(1 + 1.0/n,n);
			if(n%8 == 0){
				System.out.printf("\r%.15e\t%d",a,n);
			}
			n++;
		}
		System.out.printf("\n");
	}
}
