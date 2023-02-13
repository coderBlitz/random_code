-- Can text with nc: nc -luvz -p 2172

posix = require "posix" -- Socket things can be narrowed to posix.sys.socket


-- Open socket
server_sock = posix.socket(posix.AF_INET, posix.SOCK_DGRAM, 0)
print("Socket:", server_sock)

server_host = {family = posix.AF_INET, addr = "127.0.0.1", port = 2172}

--posix.bind(server_sock, server_host)
host = server_host
bin_dat = 1.2345
bin_dat_string = string.pack("d", bin_dat) -- string.pack needed to get binary/byte string for send/sendto
posix.sendto(server_sock, "datastring_goes_here\n", host)
posix.sendto(server_sock, bin_dat_string .. "\n", host)
print("Sent!")

-- Clean up socket
print("Cleaning")
posix.close(server_sock)
