# Experimental Rust project for Pi Pico
My simultaneous foray into embedded programming and Rust embedded programming.

# Instructions
* `cargo build`
* `arm-none-eabi-objcopy -O binary -j .text path/to/rust/binary out.bin` (probably don't want .text exclusively)
* `./uf2 path/to/rust/binary`
* Flash the resulting UF2 to pi
* Profit

# Second stage bootloader steps
Basically needs to enable XIP on the flash interface. Steps loosely in order,
 necessity may be unknown.

* Power on XIP cache (un-reset)
* Set FRF (frame format) in CTRLR0
* Enable quad I/O fast read
* Set SPI_CTRLR0_INST_L to 0
* Set SPI_CTRLR0_ADDR_L to 32 bits
* Set SPI_CTRLR0_XIP_CMD is 0xA0
* Set clock rate/divider for SSI with SCKDV in BAUDR register
* (Set) DFS_32 (CTRLR0\[20:16\]) is valid; DFS (CTRLR0\[3:0\]) is invalid
* Set TMOD in CTRLR0 (TR or TX)
* Mask all interrupts for time being?
* Disable DW_APB_SSI for configuration, enable for use??
* Enable quad mode with TRANS_TYPE and SPI_FRF (SPI_FRF only applicable if FRF is 0b00)
* SSI_SPI_MODE set to 3 for quad???
* Set WAIT_CYCLES appropriately

XIP mode (separating from above due to XIP section in datasheet)
* XIP only in dual/quad SPI mode (==> SPI_FRF != 0)
* Set SPI frame format and data frame size value in CTRLR0 (max data frame 32)
* Set address length, wait cycles, transaction type in SPI_CTRLR0 register (max address len 32)
