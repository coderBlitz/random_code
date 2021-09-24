# -*- coding: utf-8 -*-
"""
Created for Wed Sep 11 CTF
Cryptography 1 
40 Points
Welcome to the one time pad oracle! 
Our oracle's function is enc := key ^ msg | dec := key ^ ct
The oracle is found at ctf.notanexploit.club:13371, and your methods are:
    flg - returns the encrypted flag
    enc - returns the encryption of the message after the : in "enc:..."
    dec - returns the decryption of the ciphertext after the : in "dec:..."
    
@author: confusedmufasa
"""

import socket

#some potentially useful functions
def pad_equal(a,b):
    diff = len(a)-len(b)
    if diff > 0:
        b += b"\0" * diff
    else:
        a += b"\0" * -diff
    return a,b
    
def xor_bytes(a,b):
    return bytes(x ^ y for x, y in zip(a, b))


lower = [chr(n) for n in range(97,123)]

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ('ctf.notanexploit.club', 13371)
sock.connect(server_address)

#available methods: flg, enc, dec.


msg = 'flg'.encode()
sock.sendall(msg)
flg = sock.recv(1024)
print("Flag:", flg) #not decoded, because now the oracle sends encrypted bytes.

msg = 'enc:LET ME IN!!!'.encode()
sock.sendall(msg)
enc = sock.recv(1024)
print(enc)

"""
for letter in lower:
	msg = ('enc:'+letter*20).encode()
	print("Message:", msg)
	sock.sendall(msg)
	enc = sock.recv(1024)
	print("Return:", enc)

	for byte in enumerate(enc, 0):
		if byte[1] == 0:
			print("Position", byte[0], "starts with", letter)
"""
msg = ('enc:'+'\0'*15).encode()
print("Sending message:", msg, "(", len(msg), ")")

#msg = b'dec:' + enc
sock.sendall(msg)
dec = sock.recv(1024)
print(dec) #sanity check

(flg, dec) = pad_equal(flg, dec)
result = []
for i in range(len(flg)):
	c = flg[i]^dec[i]
	result.append(chr(c))

print(''.join(result))

sock.close()
