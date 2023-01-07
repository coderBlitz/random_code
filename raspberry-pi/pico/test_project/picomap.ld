MEMORY {
	/* May change to rx later if fast floating-point libs are used */
	ROM (r) : ORIGIN = 0, LENGTH = 16K

	/* All memory, addressed as contiguous block */
	SRAM (rwx) : ORIGIN = 0x20000000, LENGTH = 264K

	/* Striped memory blocks, accessed as a contiguous block */
	SRAM_STRIPED (rwx) : ORIGIN = 0x20000000, LENGTH = 256K

	/* 4k memory banks, accessed as contiguous blocks */
	SRAM_SMALL0 (rwx) : ORIGIN = 0x20040000, LENGTH = 4K
	SRAM_SMALL1 (rwx) : ORIGIN = 0x20041000, LENGTH = 4K

	/* 64k memory banks, accessed as contiguous blocks */
	SRAM_BANK0 (rwx) : ORIGIN = 0x21000000, LENGTH = 64K
	SRAM_BANK1 (rwx) : ORIGIN = 0x21010000, LENGTH = 64K
	SRAM_BANK2 (rwx) : ORIGIN = 0x21020000, LENGTH = 64K
	SRAM_BANK3 (rwx) : ORIGIN = 0x21030000, LENGTH = 64K

	/* XIP flash/cache, if the cache is disabled (clearing CTRL.EN register bit)
	The flash memory on pi pico is 2 MB, and the cache is 16 kB
	XIP registers start at 0x14000000
		* 0x00 - CTRL
			Bits 31:4 - Reserved
			Bit 3 (rw) - POWER_DOWN
			Bit 2 - Reserved
			Bit 1 (rw) - ERR_BADWRITE
			Bit 0 (rw) - EN
		* 0x04 - FLUSH
	*/
	XIP_FLASH (rwx) : ORIGIN = 0x10000000, LENGTH = 2M
	XIP_CACHE (rwx) : ORIGIN = 0x15000000, LENGTH = 16k

	/* USB DPRAM memory if the USB port is not used */
	USB_DPRAM (rwx) : ORIGIN = 0x50100000, LENGTH = 4K
}

ENTRY(_strat)
/*OUTPUT_FORMAT(binary)*/

SECTIONS {
	.text ALIGN(256) : {
		*(.text*)
	} > XIP_FLASH
	/*.data ALIGN(256) : {*/
}
