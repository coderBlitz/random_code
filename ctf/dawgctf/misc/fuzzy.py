#!/usr/bin/env python3
import random
import sys

#our wordlist.
words = ["wittiest","gimleted","awkwardest","Dalia","gangways","sideswipe",
         "portmanteau","monicker","toweling","zealot","axiom","twofold",
         "payoffs","goober","Canaan","precariously","stalemates",
         "theatrically","Venezuela","encyclopedia"];


def gen_out(seed):
    """This function generates random output based on an input seed.
    
    Parameters
    ----------
    seed
        An int that is seeded for "random" output
    
    Returns
    -------
    str
        Some "random" output
    """
    random.seed(seed);
    out = "";
    
    #the divine length
    the_chosen_length = random.randint(0,seed % 10)+10
    
    #for each length we've chosen
    for i in range(the_chosen_length):
        #find the divine chosen word
        the_chosen_word = random.choice(words);
        
        #add the divine chosen character from the chosen word
        if i == 7:
            out += "{";
        elif i == the_chosen_length-1:
            out += "}";
        elif i in (0, 4, 5, 6):
            out += the_chosen_word[random.randint(0,len(the_chosen_word)-1)].upper();
        else:
            out += the_chosen_word[random.randint(0,len(the_chosen_word)-1)].lower();
    
    return out;

#main
if __name__ == "__main__":
	if sys.version_info[0] < 3:
		raise Exception("Python 3 or a more recent version is required")
	print(gen_out(int(sys.argv[1])));

	# 3132888 has flag
	for i in range(3000000, 3200000):
		out = gen_out(i)
		if out.find("DawgCTF") != -1:
			print("Out:", out, "|", i)
