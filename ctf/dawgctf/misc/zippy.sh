#!/bin/bash

while true;do
	std=$(file flag|grep "Zip")
	gz=$(file flag|grep -i "gzip")
	bz=$(file flag|grep -i "bzip2")

	if [ "$bz" != "" ];then
		echo "Bzip2"
		cp flag flag.bz2
		bzip2 -dfk flag.bz2
	elif [ "$gz" != "" ];then
		echo "Gzip"
		cp flag flag.gz
		gzip -df flag.gz
	elif [ "$std" != "" ];then
		echo "Std zip"
		cp flag flag.zip
		unzip -o flag.zip
	else
		echo "Must be flag, exiting"
		exit 0
	fi

	if [ $? != 0 ];then
		echo "SOme error occurred"
		exit 1
	fi

	size=$(du -h flag|cut -d' ' -f1)
	echo $size
done
