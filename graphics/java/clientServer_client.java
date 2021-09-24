import java.io.*;
import java.net.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class clientServer_client extends JFrame{
	
	private JTextField userText;
	private JTextArea chatWindow;
	private ObjectOutputStream output;
	private ObjectInputStream input;
	private String message = "";
	private String serverIP;
	private Socket connection;
	
	//constructor
	public clientServer_client(String host){
		super("Client Instant Messager");
		serverIP = host;
		userText = new JTextField();
		userText.setEditable(false);
		userText.addActionListener(
					new ActionListener(){
						public void actionPerformed(ActionEvent event){
							sendMessage(event.getActionCommand());
							userText.setText("");
						}
					}
				);
		add(userText, BorderLayout.NORTH);
		chatWindow = new JTextArea();
		chatWindow.setEditable(false);
		add(new JScrollPane(chatWindow), BorderLayout.CENTER);
		setSize(700,500);
		setVisible(true);
	}
	
	//connect to server
	public void startRunning(){
		try{
			connectToServer();
			setupStreams();
			whileChatting();
		}catch(Exception e){
			showMessage("\n Client terminated connection");
			e.printStackTrace();
		}finally{
			closeCrap();
		}
	}
	
	//connect to server
	private void connectToServer() throws IOException{
		showMessage("Attempting connection... \n");
		connection = new Socket(InetAddress.getByName(serverIP),5000);
		showMessage("Connected to: " + connection.getInetAddress().getHostName());
	}
	
	//Setup streams to send and receive messages
	private void setupStreams() throws IOException{
		output = new ObjectOutputStream(connection.getOutputStream());
		output.flush();
		input = new ObjectInputStream(connection.getInputStream());
		showMessage("\n Your streams are now good to go! \n");
	}
	
	//While chatting with server
	private void whileChatting() throws IOException{
		ableToType(true);
		do{
			try{
				message = (String) input.readObject();
				showMessage("\n" + message);
			}catch(Exception e){
				showMessage("\n idk that object type ");
			}
		}while(!message.equals("SERVER: END") || !message.equals("END"));
	}

	//close the streams and sockets
	private void closeCrap(){
		ableToType(false);
		showMessage("\n closing crap down...");
		try{
			output.close();
			input.close();
			connection.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}

	//send messages to server
	private void sendMessage(String message){
		try{
			output.writeObject("CLIENT: " + message);
			output.flush();
			showMessage("\nCLIENT: " + message);
		}catch(Exception e){
			chatWindow.append("\n something messed up sending message");
		}
	}
	
	//change/update chatWindow
	private void showMessage(final String m){
		SwingUtilities.invokeLater(
				new Runnable(){
					public void run(){
						chatWindow.append(m);
					}
				}
			);
		}
	
	//gives user permission to type crap into the text box
	private void ableToType(final boolean tof){
		SwingUtilities.invokeLater(
				new Runnable(){
					public void run(){
						userText.setEditable(tof);
					}
				}
			);
		}
	
}

