package testingGrounds;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class runnables {
	
	static ThreadGroup g = new ThreadGroup("g1");
	static CyclicBarrier b = new CyclicBarrier(2);
	
	public static void main(String[] args) throws InterruptedException {
		
		System.out.print("Starting thread: ");
		test();
		
		System.out.println("Waiting for barrier to open...");
		Thread.sleep(2000);
		
		System.out.println("Opening barrier..");
		try {b.await();} catch (BrokenBarrierException e) {e.printStackTrace();}
		Thread.sleep(3000);
		
		System.out.println("Exiting..");
		System.exit(0);
	}

	public static void test() throws InterruptedException{
		Thread lol = new Thread(g,new Runnable(){
			public void run() {
				try {b.await();} catch (InterruptedException | BrokenBarrierException e1) {e1.printStackTrace();}
				int i=0;
				while(true){
					System.out.printf("i=%d\n",i);
					i++;
					try {Thread.sleep(200);} catch (InterruptedException e) {}
				}		
			}
		});
		System.out.println(lol.getName());
		Thread.sleep(1000);
		lol.start();
	}

}