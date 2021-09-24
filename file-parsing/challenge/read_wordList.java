import java.io.*;

public class read_wordList{

/* Excepts arguments passed from command line
   Format: java read_wordlist scrambledWord scrambledWord...
   Can handle any amount of argument words
*/
	public static void main(String[] args){
		boolean debug=false;

		//Prints every argument word passed
		//for(int i=0;i<args.length;i++) System.out.println("\""+args[i]+"\"");

		String fileName="wordlist.txt";
		try{
		FileReader tmp = new FileReader(fileName);
		BufferedReader reader = new BufferedReader(tmp);

		int lines=0;
		char ch;

if(debug) System.out.printf("Counting lines in file...\n");
		String s;
		while((s = reader.readLine()) != null)lines++;//Counts number of lines in the file

		System.out.printf("Counted %d lines from file \"%s\"\n",lines,fileName);
		reader.close();
		tmp.close();

		FileReader list = new FileReader(fileName);
		reader = new BufferedReader(list);
		String[] word = new String[lines];
if(debug) System.out.printf("Reading words...\n");
		for(int i=0;i<lines;i++){//Goes through file and reads every word into an array
		   word[i] = reader.readLine();
		}
		reader.close();
		list.close();

if(debug) System.out.printf("Comparing words...\n");
if(debug) System.out.printf("Arguments given: %d\n",args.length);

		for(int arg_word=0; arg_word<args.length; arg_word++){//For every argument passed
			for(int i=0;i<lines;i++){//For each word in list
				int n=0;
				for(int j=0;j<(args[arg_word].length());j++){//For each letter in argument word
					ch = args[arg_word].charAt(j);
					if(ch == '\0')break;
					for(int k=0;k<word[i].length();k++){//For each letter in word from list
						ch = word[i].charAt(k);
						if(ch == '\0') break;
						if(args[arg_word].charAt(j) == word[i].charAt(k)){
		//		   System.out.printf("Possible word for \"%c\": %s\n",args[j],word[i]);
						   n++;//Adds up number of matches in same word
						   break;
						}
					}
				}
		// If the number of matches is equal to a words length, and the argument
		// words length equals the number of matches. Resolves issue of longer words
		// containing all the letters of a shorter argument word, giving wrong word
			if((n == (args[arg_word].length())) && (n == (word[i].length()))){
				char[] word1 = new char[n];
				char[] word2 = new char[n];
				for(int j=0;j<n;j++){//gets letters in arg word and other word
					word1[j]=args[arg_word].charAt(j);
					word2[j]=word[i].charAt(j);
				}
				for(int j=0;j<n;j++){//Checks to see if every letter is used
//printf("Loop #%d: %c\n",j,word1[j]);
					for(int k=0;k<n;k++){//Loops through letters in word from list
						if(word1[j] == word2[k]){//If the letters match
//System.out.printf("Both words: %c/%d\n",word1[j],word1[j]);
							word1[j]=0;//Set both spots to 0 to
							word2[k]=0;//make checking later easier
							break;
						}
					}
				}
			int correct=1;
				for(int j=0;j<n;j++){//Goes through every letter for the word lengths
//printf("Word1[%d]: %c/%d\nWord2[%d]: %c/%d\n",j,word1[j],word1[j],j,word2[j],word2[j]);
					if(!(word1[j] == 0) || !(word2[j] == 0)){
						correct=0;//If any letters were left over, words do not match
					}
				}
			if(!(correct == 1)) continue;//If any letters are unmatched, check for another word

			System.out.printf("%s,",word[i]);
			break;
			}
		}//////////////////////////END OF ARGUMENT WORD LOOP///////////////////////////////////
		}
		System.out.printf("\n");
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
}
