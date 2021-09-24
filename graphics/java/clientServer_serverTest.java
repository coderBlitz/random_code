import javax.swing.JFrame;

public class clientServer_serverTest {

	public static void main(String[] args) {
		clientServer_server host = new clientServer_server();
		host.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		host.startRunning();
	}

}
