
public class multiThread{
	
	static Runnable thing = new Runnable(){
		public void run(){
			// Code goes here
			System.out.printf("Thread here!\nGoing back to main method..\n");
		}
	};

	public static void main(String args[]){
		Thread t = new Thread(thing);
		t.start();// Start the thread, calling run() method
		try{ t.join(); }catch(InterruptedException e){ }// Wait for it to finish

		System.out.printf("Done!\n");
	}
}
