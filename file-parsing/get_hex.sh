#!/bin/bash
#Uses get_hex to convert hex numbers to text
hexdump $1 > dumpFile.txt
./get_hex
exit
