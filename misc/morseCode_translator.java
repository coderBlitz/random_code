import java.util.Scanner;

public class morseCode_translator{

	public static void main(String[] args) {
		
		Scanner input = new Scanner(System.in);
		String message;
		int msgLength;
		String[] lookUp = new String[36];
		
	
	//International Morse Code	
	///////////Numbers 0-9 in array order////////////////////
		lookUp[0]="— — — — —";//0
		lookUp[1]="· — — — —";//1...
		lookUp[2]="· · — — —";
		lookUp[3]="· · · — —";
		lookUp[4]="· · · · —";
		lookUp[5]="· · · · ·";
		lookUp[6]="— · · · ·";
		lookUp[7]="— — · · ·";
		lookUp[8]="— — — · ·";
		lookUp[9]="— — — — ·";//9
	//////////////////////////////////////////////////////////
	///////////Characters A-Z in ascending order//////////////
		lookUp[10]="· —";//A
		lookUp[11]="— · · ·";//B
		lookUp[12]="— · — ·";//C
		lookUp[13]="— · ·";//D
		lookUp[14]="·";//E
		lookUp[15]="· · — ·";//F
		lookUp[16]="— — ·";//G
		lookUp[17]="· · · ·";//H
		lookUp[18]="· ·";//I
		lookUp[19]="· — — —";//J
		lookUp[20]="— · —";//K
		lookUp[21]="· — · ·";//L
		lookUp[22]="— —";//M
		
		lookUp[23]="— ·";//N
		lookUp[24]="— — —";//O
		lookUp[25]="· — — ·";//P
		lookUp[26]="— — · —";//Q
		lookUp[27]="· — ·";//R
		lookUp[28]="· · ·";//S
		lookUp[29]="—";//T
		lookUp[30]="· · —";//U
		lookUp[31]="· · · —";//V
		lookUp[32]="· — —";//W
		lookUp[33]="— · · —";//X
		lookUp[34]="— · — —";//Y
		lookUp[35]="— — · ·";//Z
	///////////////////////////////////////////////////////////////
		
		
		System.out.println("International Morse Code translator\nType message here (use characters A-Z,a-z and 0-9):");
		message = input.nextLine();
		message = message.toUpperCase();
		msgLength = message.length();
		
		int x=0; System.out.printf("\"");
		while (x<msgLength) {
			int a = message.charAt(x);
			if (65<=a && a<=90) {
				System.out.print(lookUp[a-55]+"   ");
			}
			else if (48<=a && a<=57){
				System.out.print(lookUp[a-48]+"   ");
			}
			else if (a == 32) {
				System.out.print("       ");
			}
			else {
				System.out.println("Invalid character");
			}
			x++;
		}
		input.close();
		System.out.println("\"");
	}

}
