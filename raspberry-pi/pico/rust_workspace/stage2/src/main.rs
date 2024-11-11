//! Pico compilation and flashing test
//!
//! Simple file to test compilation of rust code for flashing the pi pico.

#![no_std]
#![no_main]

#![allow(dead_code)]

use core::arch::asm;
use core::panic::PanicInfo;

const XIP_CTRL_BASE: u32 = 0x1400_0000;
const XIP_SSI_BASE: u32 = 0x1800_0000;
const XIP_SSI_CTRLR0: u32 = XIP_SSI_BASE;
const XIP_SSI_CTRLR1: u32 = XIP_SSI_BASE + 0x04;
const XIP_SSI_SSIENR: u32 = XIP_SSI_BASE + 0x08;
const XIP_SSI_BAUDR: u32 = XIP_SSI_BASE + 0x14;
const XIP_SSI_SPI_CTRLR0: u32 = XIP_SSI_BASE + 0xF4;

const RESETS_BASE: u32 = 0x4000_C000;
const RESET_DONE: u32 = RESETS_BASE + 0x8;

/// The entry function on boot (as defined in picomap.ld)
///
/// Will call all other functions and do all processing. Not named _start or
///  _main to avoid accidental success when linking/compiling.
//
// * Write SSI control register (INST_L = 0, ADDR_L = 32 bits, XIP_CMD = 0xa0, SPI_FRF != 0, data frame size, addr len, wait cycles, transaction type)
// * Enable/power XIP cache (may need to do this before control register)
/*
  XIP_SSI->SSIENR = 0;

  XIP_SSI->BAUDR = 2; // Must be even

  XIP_SSI->CTRLR0 = (XIP_SSI_CTRLR0_SPI_FRF_STD << XIP_SSI_CTRLR0_SPI_FRF_Pos) |
      (XIP_SSI_CTRLR0_TMOD_EEPROM_READ << XIP_SSI_CTRLR0_TMOD_Pos) |
      ((32-1) << XIP_SSI_CTRLR0_DFS_32_Pos);

  XIP_SSI->CTRLR1 = (0 << XIP_SSI_CTRLR1_NDF_Pos);

  XIP_SSI->SPI_CTRLR0 = (0x03/*READ_DATA*/ << XIP_SSI_SPI_CTRLR0_XIP_CMD_Pos) |
    ((24 / 4) << XIP_SSI_SPI_CTRLR0_ADDR_L_Pos) |
    (XIP_SSI_SPI_CTRLR0_INST_L_8B << XIP_SSI_SPI_CTRLR0_INST_L_Pos) |
    (XIP_SSI_SPI_CTRLR0_TRANS_TYPE_1C1A << XIP_SSI_SPI_CTRLR0_TRANS_TYPE_Pos);

  XIP_SSI->SSIENR = XIP_SSI_SSIENR_SSI_EN_Msk;
*/
#[no_mangle]
pub extern "C" fn _strat() -> ! {
	enable_xip()
	//xip()

	//loop {}
}

#[inline(always)]
fn xip() -> ! { unsafe {
	asm!(
		// Set stack
		"ldr r0, =0x20004000",
		"mov sp, r0",

		"ldr r0,=0x18000008",
		"ldr r1,=0x00000000",
		"str r1,[r0]",

		"ldr r0,=0x18000014",
		"ldr r1,=0x00000008",
		"str r1,[r0]",

		"ldr r0,=0x18000000",
		"ldr r1,=0x001F0300",
		"str r1,[r0]",

		"ldr r0,=0x180000F4",
		"ldr r1,=0x03000218",
		"str r1,[r0]",

		"ldr r0,=0x18000004",
		"ldr r0,=0x00000000",
		"str r1, [r0]",

		"ldr r0,=0x18000008",
		"ldr r1,=0x00000001",
		"str r1,[r0]",

		"b #0xD2",
		//"ldr r0, =0x20000100",
		//"bx.n r0",
		//"bl =0x20000100",
		options(noreturn)
	)
}}

/*#[inline(always)]
fn enable_xip() -> ! { unsafe {
	// Set stack pointer to 0x2004_1000 (whichever end is necessary)
	let saddr = 0x2004_0000;
	asm!(
		"mov sp, {saddr}",
		saddr = in(reg) saddr,
		options(nomem)
	);

	// Make register vars
	let xip_ssienr = XIP_SSI_SSIENR as *mut u32;
	let xip_ctrl = XIP_CTRL_BASE as *mut u32;
	let xip_ctrlr0 = XIP_SSI_CTRLR0 as *mut u32;
	let xip_ctrlr1 = XIP_SSI_CTRLR1 as *mut u32;
	let xip_spi_ctrlr0 = XIP_SSI_SPI_CTRLR0 as *mut u32;
	let xip_baudr = XIP_SSI_BAUDR as *mut u32;

	// TODO: FIX XIP INIT IT DOESN'T WORK.
	// TODO: BX instructions don't work either...
	// Nothing works with the flash...

	//enable_iobank();

	// Disable cache
	//xip_ctrl.write_volatile(0x0);

	// Disable SSI
	xip_ssienr.write_volatile(0);

	// Set baud rate (clock divider)
	xip_baudr.write_volatile(2);

	// Set SPI_FRF -> 0x01, DFS_32 -> 31, TMOD -> 0x3
	xip_ctrlr0.write_volatile(0x001F_0000); // TMOD really 0x3?? trying 0x0
	//xip_ctrlr0.write_volatile(0x0037_0300);
	//xip_ctrlr0.write_volatile(0x005E_0300); // Quad mode??

	// See Pg. 571 and 608
	// Set XIP_CMD -> 0x03, ADDR_L -> 0x6 (24 bits), INST_L = 0x2  (8 bits), TRANS_TYPE -> 0x0
	xip_spi_ctrlr0.write_volatile(0x0300_0218);

	// Set NDF -> 0
	//xip_ctrlr1.write_volatile(0);

	// Enable cache (should be from reset already).
	//xip_ctrl.write_volatile(0x0);

	// Enable SSI
	xip_ssienr.write_volatile(1);

	// Address for main
	let main_addr = 0x1000_0100; // Should work because jump into RAM does. Unlikely the issue.
	//let mem_addr = 0x1000_0100;
	//let addr = 0x2000_0100;
	//let addr = 0x0100_1000;
	//let addr = 0x0001_0010;

	/*let prog = mem_addr as *const u32;
	let mem = main_addr as *mut u32;

	for i in 0..0x200 {
		mem.add(i).write_volatile(prog.add(i).read_volatile());
	}*/

	// Test read.
	/*asm!(
		"mov {r}, {addr}",
		"ldr {r}, [{r}, #0]",
		r = out(reg) _,
		addr = in(reg) main_addr,
	);*/

	// Jump to main.
	asm!(
		"mov pc, {addr}",
		addr = in(reg) main_addr,
		options(noreturn)
	)
}}*/

/// Yet another attempt to get XIP working. Refer to SPI_notes.md for details.
///
/// The process this time:
/// 1. Check PSM_BASE.DONE register for all needed peripheral's registers.
/// 2. Un-reset RESETS_BASE for any needed peripherals (SPI, IO, etc.). (TODO: check if this is always before config)
/// 3. Check RESETS_DONE register for ready status.
/// 4. Write SSI control register (INST_L = 8 bits, ADDR_L = 24 bits, XIP_CMD = 0x03). May need to disable SSI before config.
/// 4b. After XIP enabled (should verify with testing first), enable quad read with (INST_L = 0, ADDR_L = 32 bits, XIP_CMD = 0xa0)
/// 5. XIP
// TODO: Try inserting NOP delays or find status registers to check.
#[inline(always)]
fn enable_xip() -> ! { unsafe {
	// Make register vars
	let resets = RESETS_BASE as *mut u32;
	let resets_done = RESET_DONE as *mut u32;
	let xip_ssienr = XIP_SSI_SSIENR as *mut u32;
	let xip_ctrl = XIP_CTRL_BASE as *mut u32;
	let xip_ctrlr0 = XIP_SSI_CTRLR0 as *mut u32;
	let xip_ctrlr1 = XIP_SSI_CTRLR1 as *mut u32;
	let xip_spi_ctrlr0 = XIP_SSI_SPI_CTRLR0 as *mut u32;
	let xip_baudr = XIP_SSI_BAUDR as *mut u32;

	// Quad-mode (0x2), data frame size 32 bits (0x1F)
	const CTRLR0: u32 = 0 | 0x2 << 21 | 0x1F << 16;
	//
	// See Pg. 571 and 608
	// XIP_CMD = 0xa0, WAIT_CYCLES = 31/32 (0x1F) from DFS_32, INST_L = 0, ADDR_L = 32 bits ()
	const SPI_CTRLR0: u32 = 0 | 0xa0 << 24 | 0x1F << 11 | 0x8 << 2;

	// TODO: Check PSM and RESETS_DONE registers.

	// Un-reset things
	//const RESETS: u32 = !0 ^ (1 << 17 | 1 << 16 | 1 << 6 | 1 << 5);
	const RESETS: u32 = 0; // Go for broke
	resets.write_volatile(RESETS);

	//while resets_done.read_volatile() != RESETS {}
	for _ in 0..9000 {
		nop();
	}

	// Disable SSI
	xip_ssienr.write_volatile(0);

	// Enable cache
	xip_ctrl.write_volatile(0x1);

	// TODO: Delay or check register status??
	for _ in 0..9000 {
		nop();
	}

	// Set baud rate (clock divider)
	xip_baudr.write_volatile(4);

	// Enable XIP with value described above.
	xip_ctrlr0.write_volatile(CTRLR0);
	xip_spi_ctrlr0.write_volatile(SPI_CTRLR0);

	// Set NDF -> 0
	xip_ctrlr1.write_volatile(0);

	// Enable SSI
	xip_ssienr.write_volatile(1);

	// Enable cache (again)
	xip_ctrl.write_volatile(0x1);

	for _ in 0..9000 {
		nop();
	}

	// Address for main
	let main_addr = 0x1000_0100; // Should work because jump into RAM does. Unlikely the issue.

	// Test read.
	asm!(
		"mov {r}, {addr}",
		"ldr {r}, [{r}, #0]",
		r = out(reg) _,
		addr = in(reg) main_addr,
	);

	// Set stack address
	let saddr = 0x2004_0000;
	asm!(
		"mov sp, {saddr}",
		saddr = in(reg) saddr,
		options(nomem)
	);

	for _ in 0..9000 {
		nop();
	}

	// Jump to main.
	asm!(
		"mov pc, {addr}",
		addr = in(reg) main_addr,
		options(noreturn)
	)
}}

#[inline(always)]
fn enable_iobank() {
	let resets = RESETS_BASE as *mut u32;
	let resets_status = RESET_DONE as *mut u32;

	// (1.) Write to reset register to enable IO_BANK0 (for GPIO), then wait for ready
	let reset_state = 0xFFFF_FFDF;
	unsafe { resets.write_volatile(reset_state); }
	while unsafe { resets_status.read_volatile() & 0x20 } == 0x0 {}
}

#[inline(always)]
fn nop() {
	unsafe {
		asm!("NOP", options(nomem, nostack));
	}
}

#[panic_handler]
fn panic( _info: &PanicInfo) -> ! {
	loop {}
}
