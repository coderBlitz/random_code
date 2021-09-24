<?php
/*This is a word sorting program for HackThisSite.org programming mission #1
Supposed to take the scrambled words
*/
/*if(argc > 1){
	for(int $i=1;$i<$argc;i++) printf("\"%s\" ",$argv[i]);
	printf("\n");
}*/

$fileName="wordlist.txt";
$fp = fopen($fileName,"r");

$lines=0;

//Counts number of lines, by extension number of words
while(($ch = fgets($fp)) != false) $lines++;


printf("Counted %d lines from file \"%s\"\n",$lines,$fileName);
rewind($fp);

$word = array_fill(0,$lines,'');// Initialize array with 'lines' elements
for($i=0;$i<$lines;$i++){//Puts words from file into an array
	$line = fgets($fp);
	$word[$i] = substr($line,0,-2);
}


for($arg_word=1; $arg_word<$argc; $arg_word++){//For every word passed as argument
	for($i=0;$i<$lines;$i++){//For each word in list
		$n=0;

		for($j=0;$j<10;$j++){//For each letter in argument word
			if($j >= strlen($argv[$arg_word])) break;
			$ch = $argv[$arg_word][$j];
			if($ch == '\0')break;
			for($k=0;$k<10;$k++){//For each letter in word from list
				if($k >= strlen($word[$i])) break;
				$ch = $word[$i][$k];
				if($ch == '\0') break;

				if($argv[$arg_word][$j] == $word[$i][$k]){
	//			   printf("Possible word for \"%s\": %s\n",$argv[$arg_word],$word[$i]);
				   $n++;//Adds up number of matches in same word
				   break;
				}
			}
		}

	//If the number of matches is equal to a words length, and the argument
	//words length equals the number of matches. Resolves issue of longer words
	//containing all the le;tters of a shorter argument word, giving wrong word
		if(($n == strlen($argv[$arg_word])) && ($n == strlen($word[$i]))){
			$word1 = array_fill(0,$n,'');
			$word2 = array_fill(0,$n,'');
			for($j=0;$j<$n;$j++){//gets letters in arg word and other word
				$word1[$j]=$argv[$arg_word][$j];
				$word2[$j]=$word[$i][$j];
			}
			for($j=0;$j<$n;$j++){//Checks to see if every letter is used
//printf("Loop #%d: %s\n",$j,$word1[$j]);
				for($k=0;$k<$n;$k++){//Loops through letters in word from list
					if($word1[$j] == $word2[$k]){
//						printf("Both words: %s/%d\n",$word1[$j],ord($word1[$j]));
						$word1[$j]='\0';//Makes checking later easier
						$word2[$k]='\0';
						break;
					}
				}
			}
			$correct=true;
			for($j=0;$j<$n;$j++){//Goes through every letter in the words
//printf("Word1[%d]: %s/%d\nWord2[%d]: %s/%d\n",$j,$word1[$j],ord($word1[$j]),$j,$word2[$j],ord($word2[$j]));
				if(!($word1[$j] == '\0') || !($word2[$j] == '\0')){
					$correct=false;//If any letters were left over, words do not match
				}
			}
			if(!$correct) continue;//If any letters are unmatched, check for another word

			printf("%s,",$word[$i]);
		break;
		}
	}
}
fclose($fp);

//This is for if the characters mess up somewhere. Shows content of a word to represent all word contents
/*for($i=0;$i<10;$i++){
	if($i >= strlen($word[1227])) break;
	printf("Word 1227[%d]: %s/%d\n",$i,$word[1227][$i],ord($word[1227][$i]));
} */

printf("\n");

?>
