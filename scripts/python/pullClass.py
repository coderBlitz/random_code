#!/bin/env python3

# Trying to mimick pullClass.sh in python, for easier processing

from datetime import datetime
import http.client
import sys

BASEURL = "highpoint-prd.ps.umbc.edu"
COOKIEPAGE="/app/catalog/classSearch"
DBURL="/app/catalog/getClassSearch"

c = http.client.HTTPSConnection(BASEURL, timeout=10)

c.request("GET", COOKIEPAGE)
res = c.getresponse()

if res.status != 200:
	print("Request failed:", res.status, res.reason)
	sys.exit(-1)

cookiestring = res.getheader("Set-Cookie")
cookies = cookiestring.split(';')
#c.close()
#print("Cookies:", cookies)

for cookie in cookies:
	parts = cookie.split('=')
	if len(parts) >= 2:
		if parts[0] == "CSRFCookie":	
			
			token = parts[1]
		if parts[0] == "expires":
			expiration = datetime.strptime(parts[1], "%a, %d-%b-%Y %H:%M:%S %Z")
			print("Token expires:", expiration)

			f = open("/tmp/pullClass-token")
			f.write(bytearray(token, "UTF-8"))
			f.write('\n')
			f.write(expiration)
			f.close()

		print(parts[0], ": ", parts[1], sep='')

term=2198
subject="CMSC"
params = "CSRFToken="+token+"&term="+str(term)+"&subject="+subject
print(params)
headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "application/x-www-form-urlencoded"}

#c = http.client.HTTPSConnection(BASEURL, timeout=10)
thing = bytes(params, "ASCII")
c.request("POST", DBURL, thing, headers)

res = c.getresponse()

while res.status == 302:
	url = res.getheader("Location")
	print("Redirecting to '", url, "'", sep='')
	c.request("GET", url)
	res = c.getresponse()

data = res.read()
f = open("/tmp/page.html", "wb")
f.write(data)
f.close()

if res.status != 200:
	print("Class request failed:", res.status, res.reason, res.getheaders())
	print("Location is", )
	sys.exit(-1)


c.close()
