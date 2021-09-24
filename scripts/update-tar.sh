#!/bin/bash
# Update tar archive in place including new files and deleted ones.
# The tar '--append' and '--update' options add the file again at the end
# of the archive, making it bigger the more changes are made. This script
# goes through all the files currently in the archive and checks if they have
# been updated or deleted. If the file being checked is a directory, then the
# script checks for new files to add to the archive.

# echo $1
# If the file parameter exists
if [ -f "$1" ];then
	TAR_FILE="$1"
# If it is an uncompressed tar archive
	if [ "${TAR_FILE##*.}" != "tar" ];then
		echo "Please make sure the tar file exists and is uncompressed"
		exit 1
	fi
else
	echo "Tar file not exists"
	exit 1
fi

# List will be all the files present in the archive
list=($(tar tvf $TAR_FILE|sed 's/  */ /g'|cut -d' ' -f6))

for i in ${list[*]};do
#echo ${list[*]}
# If the file does not exist, remove it from archive
	stat "$i" &> /dev/null
	if [ "$?" == "1" ];then
		echo "Removing '$i' from archive.."
		tar f $TAR_FILE --delete "$i" &>/dev/null
	fi

# If the file changed, delete then re-add as to not repeat files.
	tar df $TAR_FILE "$i" &> /dev/null
	if [ "$?" == "1" ];then
# If the file is a directory, skip for later
		if [ ! -d "$i" ];then
			echo "Updating '$i'"
			tar f $TAR_FILE --delete "$i" &> /dev/null
			tar rf $TAR_FILE "$i"
		fi
	fi

# If the file is a directory then check for new files
	if [ -d "$i" ];then
		tmp=($(find "$i" -printf "%p\n"))
# For every file in the directory
		for file in ${tmp[*]};do
# Check is the file in the directory in the current archive
			rs=$(echo ${list[*]}|grep "$file")
			if [ "$rs" == "" ];then
				echo "Adding '$file' to archive.."
				tar rf $TAR_FILE $file
			fi
		done
	fi

done
