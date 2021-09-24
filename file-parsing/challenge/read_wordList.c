#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/*This is a word sorting program for HackThisSite.org programming mission #1
Supposed to take the scrambled words
*/

int main(int argc, char **argv){
/*if(argc > 1){
	for(int i=1;i<argc;i++) printf("\"%s\" ",argv[i]);
	printf("\n");
}*/
char fileName[21]="wordlist.txt";
FILE *fp = fopen(fileName,"r");

int lines=0;

char ch;
while(!feof(fp)){//Counts number of lines, by extension number of words
  ch = fgetc(fp);
  if(ch == '\n'){
    lines++;
  }
}
printf("Counted %d lines from file \"%s\"\n",lines,fileName);
rewind(fp);

char word[lines][10];
for(int i=0;i<lines;i++){//Puts words from file into an array
   for(int j=0;j<10;j++){//Loops through letters in each word. All are 9 or less characters so far
	char c = fgetc(fp);
	if(c == 13){
	  word[i][j] = '\0';
	}else if(c == 10){
	  word[i][j] = '\0';
	  break;
	}else word[i][j] = c;
   }
}

FILE *out = fopen("/tmp/file.txt","w");

for(int arg_word=1; arg_word<argc; arg_word++){//For every word passed as argument
	for(int i=0;i<lines;i++){//For each word in list
		int n=0;
		for(int j=0;j<10;j++){//For each letter in argument word
			ch = argv[arg_word][j];
			if(ch == '\0')break;
			for(int k=0;k<10;k++){//For each letter in word from list
				ch = word[i][k];
				if(ch == '\0') break;
				if(argv[arg_word][j] == word[i][k]){
	//			   printf("Possible word for \"%c\": %s\n",arg[j],word[i]);
				   n++;//Adds up number of matches in same word
				   break;
				}
			}
		}
	//If the number of matches is equal to a words length, and the argument
	//words length equals the number of matches. Resolves issue of longer words
	//containing all the letters of a shorter argument word, giving wrong word
		if((n == strlen(argv[arg_word])) && (n == strlen(word[i]))){
			char word1[n],word2[n];
			for(int j=0;j<n;j++){//gets letters in arg word and other word
				word1[j]=argv[arg_word][j];
				word2[j]=word[i][j];
			}
			for(int j=0;j<n;j++){//Checks to see if every letter is used
//printf("Loop #%d: %c\n",j,word1[j]);
				for(int k=0;k<n;k++){//Loops through letters in word from list
					if(word1[j] == word2[k]){
//						printf("Both words: %c/%d\n",word1[j],word1[j]);
						word1[j]=0;//Makes checking later easier
						word2[k]=0;
						break;
					}
				}
			}
			int correct=1;
			for(int j=0;j<n;j++){//Goes through every letter in the words
//printf("Word1[%d]: %c/%d\nWord2[%d]: %c/%d\n",j,word1[j],word1[j],j,word2[j],word2[j]);
				if(!(word1[j] == 0) || !(word2[j] == 0)){
					correct=0;//If any letters were left over, words do not match
				}
			}
			if(!correct) continue;//If any letters are unmatched, check for another word

			printf("%s,",word[i]);
			fprintf(out,"%s",word[i]);
		break;
		}
	}
}
fclose(fp);
fclose(out);

//This is for if the characters mess up somewhere. Shows content of a word to represent all word contents
//for(int i=0;i<10;i++) printf("Word 1227[%d]: %c/%d\n",i,word[1227][i],word[1227][i]);

printf("\n");
}

