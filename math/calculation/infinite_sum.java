public class infinite_sum{
	public static void main(String args[]){
		double sum = 0;
		long count = 1;
		while(true){
			sum += 1.0/count;
			count *= 2;
			System.out.printf("\r%f",sum);
		}
	}
}
