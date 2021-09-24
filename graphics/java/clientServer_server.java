import java.io.*;
import java.net.*;
import javax.swing.*;
import java.awt.BorderLayout;
import java.awt.event.*;

public class clientServer_server extends JFrame{
	
	static boolean running = false;
	static ServerSocket serverSocket;
	static Socket clientSocket;
	static JTextField userText;
	static JTextArea chatWindow;
	static ObjectOutputStream output;
	static ObjectInputStream input;
	
		//constructor
		public clientServer_server(){
			super("Instant Messenger Host");
			userText = new JTextField();
			userText.setEditable(false);
			userText.addActionListener(
					new ActionListener(){
						public void actionPerformed(ActionEvent e){
							sendMessage(e.getActionCommand());
							userText.setText("");
							
							}
						}
					);
			add(userText, BorderLayout.NORTH);
			chatWindow = new JTextArea();
			chatWindow.setEditable(false);
			add(new JScrollPane(chatWindow));
			setSize(700,500);
			setVisible(true);
		}
		
		//Start and run the server on Port:5000
		public void startRunning(){
			try{
				serverSocket = new ServerSocket(5000,10);
				while(true){
					try{
						waitForConnection();
						setupStreams();
						whileChatting();
					}catch(Exception e){
						e.printStackTrace();
						System.out.println("\nServer ended connection! ");
					}
				}
			}catch(Exception e){
				e.printStackTrace();
			}finally{
				stopServer();
			}			
		}
			
		//Wait for connection,then display connection info
		private void waitForConnection() throws IOException{
			showMessage("\nWaiting for someone to connect...\n");
			clientSocket = serverSocket.accept();
			showMessage("Now connected to "+clientSocket.getInetAddress().getHostName());
		}
		
		//Get stream to send and receive data
		private void setupStreams() throws IOException{
			output = new ObjectOutputStream(clientSocket.getOutputStream());
			output.flush();
			input = new ObjectInputStream(clientSocket.getInputStream());
			showMessage("\n Streams are now setup! \n");
		}
		
		//During the chat convo
		private void whileChatting() throws IOException{
			String message = " Your are now connected! ";
			sendMessage(message);
			ableToType(true);
			do{
				try{
					message = (String) input.readObject();
					showMessage("\n"+message);
				}catch(Exception e){
					showMessage("\n idk wtf that user sent!");
				}
			}while(!message.equals("CLIENT: END"));
		}
		
		//Close streams and sockets when done chatting
		private void stopServer(){
			showMessage("\n Closing connections... \n");
			ableToType(false);
			try{
				output.close();
				input.close();
				clientSocket.close();
			}catch(Exception e){
				e.printStackTrace();
			}
		}
		
		//Send a message to client
		private void sendMessage(String message){
			try{
				output.writeObject("SERVER: " + message);
				output.flush();
				showMessage("\nSERVER: " + message);
			}catch(IOException ioException){
				chatWindow.append("\n ERROR: DUDE I CANT SEND THAT MESSAGE");
			}
		}
		
		//Updates chatWindow
		private void showMessage(final String text){
			SwingUtilities.invokeLater(
						new Runnable(){
							public void run(){
								chatWindow.append(text);
							}
						}
					);
		}
		
		//let the user type stuff into their box
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
