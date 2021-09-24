package testingGrounds;

import java.util.Scanner;
import java.util.Random;

public class coinFlip {

	public static void main(String[] args) {

		Scanner input = new Scanner(System.in);
		Random number = new Random();

		int b,a=0;
		String c;
		b = number.nextInt(2);

		System.out.println("Heads or Tails?");
		c = input.next();
		if (c.equals("Heads")||c.equals("heads")||c.equals("h")) {a=1;}
		else {a=0;}
		
		if (a==b) {
			System.out.println("Correct!\n");
		  }
		else {
			System.out.println("Incorrect!\n");
		  }
		
		while (!c.equals("done")) {
		
		b = number.nextInt(2);
		
		System.out.println("Type \"done\" to stop");
		System.out.println("Heads or Tails?");
		c = input.next();
		if (c.equals("Heads")||c.equals("heads")||c.equals("h")) {a=1;}
		else if (c.equals("done")) {break;}
		else {a=0;}
		
		if (a==b) {
			System.out.println("Correct!\n");
		  }
		else {
			System.out.println("Incorrect!\n");
		  }
		}
		input.close();
	}
}
