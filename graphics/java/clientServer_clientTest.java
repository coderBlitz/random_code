import javax.swing.JFrame;

public class clientServer_clientTest {

	public static void main(String[] args) {
		clientServer_client someone;
		someone = new clientServer_client("127.0.0.1");
		someone.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		someone.startRunning();
	}

}
