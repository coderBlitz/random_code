import java.io.PrintStream;

public class perfectNumbers{
	public static void main(String[] args){
		int p=2;
		int MAX=31;
	
		while(p<=MAX){
			long n = (long)Math.pow(2,p)-1;
			long f=0;
			printf("\rCurrent value of p: %d",p);

			for(long j=1;j<=n;j++){
				if(n%j == 0){f += j;}
			}
			if(f==n+1){
				n *= Math.pow(2,p-1);
				printf("\r%d is a Perfect Number\n",n);
			}
			if(p==1 || p==2) p++;
			else p+=2;
		}	
		System.exit(0);
	}
	static PrintStream printf(String fmt,Object...args){
		return System.out.printf(fmt,args);
	}
}
