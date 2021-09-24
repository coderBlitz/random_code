#!/bin/bash

#DEVICE=/dev/sdc2
echo -n "Enter the device: "
read DEVICE

gpg -o - usb.gpg|cryptsetup --key-file=- luksOpen $DEVICE usb
sudo mount /dev/mapper/usb ~/mnt/
