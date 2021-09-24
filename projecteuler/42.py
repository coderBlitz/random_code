#	By converting each letter in a word to a number corresponding to its
#	 alphabetical position and adding these values we form a word value.
#	For example, the word value for SKY is 19 + 11 + 25 = 55 = t_10. If the
#	 word value is a triangle number then we shall call the word a triangle word.

#	Using words.txt (right click and 'Save Link/Target As...'), a 16K text file
#	 containing nearly two-thousand common English words, how many are triangle words?

from math import ceil,floor,sqrt

"""	wtoi -- Word to integer
"""
def wtoi(word):
	word = word.upper()
	ret = 0
	for i in word:
		ret += ord(i) - ord('A') + 1
	return ret

words = []
with open("42.txt","r") as fp:
	words = fp.readline().replace('"', '').split(',')

#print(words)
print(len(words))

count = 0
for word in words:
	val = wtoi(word)

	n = (sqrt(1 + 8 * val) - 1) / 2

	if ceil(n) == floor(n):
		count += 1

print(count)
