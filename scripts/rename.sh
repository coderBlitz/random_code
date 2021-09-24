#!/bin/bash

# Program will change file extentions in current directory
for i in *.m4v; # Extension to find "*.[extension]"
	do
#	echo ${i:(-3)};
	echo "Moving $i to ${i:0:(-4)}.mov"; # Extension to replace with "*.[newExtension]"
#	mv "$i" "${i:0:(-4)}.mov"
#	echo "${i:0:((${#i}-4))}";
#	echo ${#i}
done
