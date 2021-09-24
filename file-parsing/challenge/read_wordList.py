import sys

#This took so many Google searches to get right. And about 4 hours time. It's 01:10 right now

debug = False #'True' or 'False'

#Prints every argument word passed
#for(int i=0;i<args.length;i++) System.out.println("\""+args[i]+"\"");

fileName='wordlist.txt'
#try:
for x in range(1):
	f = open(fileName,'r')

	lines=0

	if(debug):
		print "Counting lines in file..."

	s = f.readline()
	while(s != ''):
		lines += 1 #Counts number of lines in the file
		s = f.readline()

	sys.stdout.write("Counted %d lines from file \"%s\"\n"%(lines,fileName))

	f.seek(0)
	word = []
	if(debug):
		print "Reading words..."
	for i in range(lines):#Goes through file and reads every word into an array
		word.append(f.readline())


	if(debug):
		print "Comparing words..."
	if(debug):
		print "Arguments given: %d"%len(sys.argv)

	for arg_word in range(1,len(sys.argv)): #For every argument passed
		for i in range(lines): #For each word in list
			n=0
			for j in range(len(sys.argv[arg_word])): #For each letter in argument word
				ch = sys.argv[arg_word][j]
				if(ch == '\n'):
					break
				for k in range(len(word[i])): #For each letter in word from list
					ch = word[i][k]
					if(ch == '\n'):
						break
					if(sys.argv[arg_word][j] == word[i][k]):
			  			#print "Possible word for \"%c\": %s"%(sys.argv[arg_word][j],word[i])
						n += 1 #Adds up number of matches in same word
						break
		#If the number of matches is equal to a words length, and the argument
		#words length equals the number of matches. Resolves issue of longer words
		#containing all the letters of a shorter argument word, giving wrong word
			#print "N is %d\nlength of argv[%d]: %d\nLength of word[%d]: %d"%(n,arg_word,len(sys.argv[arg_word]),i,len(word[i])-2)
			if n == (len(sys.argv[arg_word])) and n == (len(word[i])-2):
				word1 = []
				word2 = []
				for j in range(n): #gets letters in arg word and other word
					word1.append(sys.argv[arg_word][j])
					word2.append(word[i][j])
				for j in range(n): #Checks to see if every letter is used
					#print "Loop #%d: %c"%(j,word1[j])
					for k in range(n): #Loops through letters in word from list
						if(word1[j] == word2[k]): #If the letters match
							#print "Both words: %c/%d"%(word1[j],word1[j])
							word1[j]=0 #Set both spots to 0 to
							word2[k]=0 #make checking later easier
							break
				correct=1
				for j in range(n): #Goes through every letter for the word lengths
					#print "Word1[%d]: %c/%d\nWord2[%d]: %c/%d"%(j,word1[j],word1[j],j,word2[j],word2[j])
					if word1[j] != 0 or word2[j] != 0:
						correct=0 #If any letters were left over, words do not match
				if correct != 1:
					continue #If any letters are unmatched, check for another word

				sys.stdout.write("%s,"%word[i][:-2])
				break
		#//////////////////////////END OF ARGUMENT WORD LOOP///////////////////////////////////
	print
#except:
	#print "Something happened"
