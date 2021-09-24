#!/bin/bash

if [ $# -lt 1 ];then
  echo "Cant chroot to nothing"
  exit 1
fi

if [ ! -d "$1" ];then
  echo "Dont exist"
  exit 1
fi


for file in {dev,proc,run,sys};do
  mount --bind /$file "$1/$file"
done

chroot "$1"


for file in {dev,proc,run,sys};do
  umount "$1/$file"
done
