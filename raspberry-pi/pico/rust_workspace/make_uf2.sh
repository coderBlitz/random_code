#!/bin/bash

if [[ $1 == "" ]]; then
	echo "Please give the path to the main binary."
	exit 1
fi

STAGE2="target/thumbv6m-none-eabi/release/stage2"

arm-none-eabi-objcopy -O binary -j .text $STAGE2 /tmp/stage2.bin
arm-none-eabi-objcopy -O binary $1 /tmp/main.bin

cp /tmp/stage2.bin /tmp/padded.bin
truncate -s 256 /tmp/padded.bin
cat /tmp/padded.bin /tmp/main.bin > /tmp/combo.bin

./uf2 /tmp/combo.bin /tmp/combo.uf2
