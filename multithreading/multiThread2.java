package testingGrounds;

import java.io.PrintStream;
import java.lang.reflect.Field;
import java.util.Scanner;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class multiThread2 {

	static Runnable SUM = new Runnable(){
		public void run(){
			try {bar.await();} catch (Exception e) {}
			int id = (int)Thread.currentThread().getId();
			id -= 10;
			printf("Thread id: %d\n",id);
			try{total[id]=array1[id]+array2[id];} catch(ArrayIndexOutOfBoundsException e) {printf("The thread id is too high for the specified indecies\n");}
			try {bar.await();} catch (Exception e) {printf("Thread failed to catch barrier on finish\n");}
		}
	};
	
	static int[] array1;
	static int[] array2;
	static int[] total;
	static CyclicBarrier bar;
	
	multiThread(){
		Thread t = new Thread(SUM);
		printf("Thread with ID %d has been created\n",t.getId());
		t.start();
	}
	public static void main(String[] args){
		
		int N=4;
		array1 = new int[N];
		array2 = new int[N];
		total = new int[N];
		for(int b=0;b<array1.length;b++){
			array1[b]=b;
			array2[b]=b+1;
		}
		bar = new CyclicBarrier(N);
		
		for(int i=0;i<N;i++){
			multiThread thr = new multiThread();
		}
		printf("Starting threads..\n");
		try {bar.await();} catch (InterruptedException | BrokenBarrierException e){printf("Something happened\n");}
		printf("Waiting for threads to finish...\n");
		
		bar.reset();
		
		try {bar.await();}catch (InterruptedException | BrokenBarrierException e) {}
		
		printf("Printing totals..\n");
		for(int i=0;i<total.length;i++){
			printf("The sum for array1[%d] and array2[%d] is %d\n",i,i,total[i]);
		}
	}
	
	public static PrintStream printf(String format,Object...args){
		return System.out.printf(format, args);
	}
}
