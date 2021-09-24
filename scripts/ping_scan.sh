#!/bin/bash
# Scan network in 172.16.34.*
# Could be extended

# Net is the first three sets of numbers in IP address. ex 192.168.0
net=10.223.55

for i in {1..255..1} #This is the range {START..END..1}
do
ping -q -c1 -W1 $net.$i > /dev/null
success=$?
if [ "$success" == "0" ]
  then
    echo "$net.$i is active"
fi
done

