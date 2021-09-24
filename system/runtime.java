import java.io.*;

public class runtime{
	public static void main(String args[])throws IOException{// exec() and readLine() can throw IOException
		Runtime rt = Runtime.getRuntime();
		Process proc = rt.exec(new String[]{"./tmp"});// First string is program, everything after is args

		BufferedReader stdIn = new BufferedReader(new InputStreamReader(proc.getInputStream()));

		String s = null;
		while((s = stdIn.readLine()) != null) System.out.println(s);
	}
}
