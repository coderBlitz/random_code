a = raw_input("Enter the hex number to convert: ")

t=0
f=16*(len(a)-1)#This is so the correct character is multiplied by the right value
for x in range(len(a)):#The ord() method gives the decimal value of a character
	if ord(a[x]) >= 97 and ord(a[x]) <= 122:
		b = ord(a[x]) - 87
	else:
		b = ord(a[x]) - 48
	if f == 0:
		t += b
	else:
		t += b*f
	f -= 16
print t
