import os
import socket

def client(host, port):
	print("To conn to '",host,"' on port", port)

	# Create socket (can be same on client and server)
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) # Family, type, proto, filedes

	try:
		con = sock.connect((addr, port))

		msg = "Hello world!"
		data = bytes(msg, "ascii")
		sock.send(data)
	except OSError as msg:
		print("Connection failure")
		print(msg)
	except BaseException as e:
		print("Failure:", e)
		pass

	# Cleanup
	sock.close()

	os._exit(0)

def server(addr, port, N=1):
	print("To bind to", addr, "on", port)

	# Create socket (can be same on client and server)
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) # Family, type, proto, filedes

	# Bind to addres/port
	#if addr == "0.0.0.0":
	#	bindaddr = ""
	#else:
	#	bindaddr = socket.inet_aton(addr)
	#bindport = socket.htons(port)
	sock.bind((addr, port)) # AF_INET is (addr, port) tuple

	# Setup connection queue
	sock.listen(N)

	try:
		recip = sock.accept()
		conn = recip[0]
		dest = recip[1]
		print("Connected to", dest)

		print("Receiving data..")
		bufsize = 128
		data = conn.recv(bufsize)
		print("Received:", data)
	except KeyboardInterrupt:
		pass
	except BaseException as e:
		print("Server Error:", e)

	sock.close()


addr = "127.0.0.1"
host = "localhost"
port = 2000
pid = os.fork()
if pid == 0:
	client(host, port)
else:
	server(addr, port)

status = os.waitpid(pid, 0)
print(status)
print("Done")
