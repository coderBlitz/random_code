//! Pico compilation and flashing test
//!
//! Simple file to test compilation of rust code for flashing the pi pico.

#![no_std]
#![no_main]

#![allow(dead_code)]

use core::arch::asm;
use core::panic::PanicInfo;

const XIP_SSI_BASE: u32 = 0x1800_0000;
const XIP_SSI_CTRLR0: u32 = XIP_SSI_BASE;
const XIP_SSI_CTRLR1: u32 = XIP_SSI_BASE + 0x04;
const XIP_SSI_SSIENR: u32 = XIP_SSI_BASE + 0x08;
const XIP_SSI_BAUDR: u32 = XIP_SSI_BASE + 0x14;
const XIP_SSI_SPI_CTRLR0: u32 = XIP_SSI_BASE + 0xf4;

const RESETS_BASE: u32 = 0x4000_C000;

/// Resets register
///
/// All bits default to 0x1 for the reset state. All registers are RW.
///
/// * 31:25 - Reserved
/// * 24 - USBCTRL
/// * 23 - UART1
/// * 22 - UART0
/// * 21 - TIMER
/// * 20 - TBMAN
/// * 19 - SYSINFO
/// * 18 - SYSCFG
/// * 17 - SPI1
/// * 16 - SPI0
/// * 15 - RTC
/// * 14 - PWM
/// * 13 - PLL_USB
/// * 12 - PLL_SYS
/// * 11 - PIO1
/// * 10 - PIO0
/// *  9 - PADS_QSPI
/// *  8 - PADS_BANK0
/// *  7 - JTAG
/// *  6 - IO_SQPI
/// *  5 - IO_BANK0
/// *  4 - I2C1
/// *  3 - I2C0
/// *  2 - DMA
/// *  1 - BUSCTRL
/// *  0 - ADC
const RESETS_RESET: u32 = RESETS_BASE + 0x0;
const RESETS_WDSEL: u32 = RESETS_BASE + 0x4;
const RESETS_RESET_DONE: u32 = RESETS_BASE + 0x8;

const PSM_BASE: u32 = 0x4001_0000;
const PSM_FRCE_ON: u32 = PSM_BASE + 0x0;
const PSM_FRCE_OFF: u32 = PSM_BASE + 0x4;
const PSM_WDSEL: u32 = PSM_BASE + 0x8;
const PSM_DONE: u32 = PSM_BASE + 0xC;

const IO_BANK0_BASE: u32 = 0x4001_4000;
const GPIO0_STATUS: u32 = IO_BANK0_BASE + 0x000;
const GPIO0_CTRL: u32 = IO_BANK0_BASE + 0x004;

const GPIO25_STATUS: u32 = IO_BANK0_BASE + 0x0C8; // LED pin
const GPIO25_CTRL: u32 = IO_BANK0_BASE + 0x0CC;

const PADS_BANK0_BASE: u32 = 0x4001_C000;
const PADS_BANK0_GPIO25: u32 = PADS_BANK0_BASE + 0x68;

const GPIO_F5: u32 = 0x5;
const GPIO_FUNC_SIO: u32 = GPIO_F5;
const GPIO_FUNC_NULL: u32 = 0x1F;

/// The entry function on boot (as defined in picomap.ld)
///
/// Will call all other functions and do all processing. Not named _start or
///  _main to avoid accidental success when linking/compiling.
#[no_mangle]
pub extern "C" fn _strat() -> ! {
	flash_led()
}

fn enable_iobank() {
	let resets = RESETS_BASE as *mut u32;
	let resets_status = RESETS_RESET_DONE as *mut u32;

	// (1.) Write to reset register to enable IO_BANK0 (for GPIO), then wait for ready
	let reset_state = 0xFFFF_FFDF;
	unsafe { resets.write_volatile(reset_state); }
	while unsafe { resets_status.read_volatile() & 0x20 } == 0x0 {}
}

#[inline(always)]
fn set_led(on: bool) {
	let pin25_ctrl = GPIO25_CTRL as *mut u32;
	const LED_PIN_STATE_ON: u32 = 0x0000_3305;
	const LED_PIN_STATE_OFF: u32 = 0x0000_3205;

	if on {
		unsafe { pin25_ctrl.write_volatile(LED_PIN_STATE_ON); }
	} else {
		unsafe { pin25_ctrl.write_volatile(LED_PIN_STATE_OFF); }
	}
}

#[inline(always)]
fn flash_led() -> ! {
	/* Attempt to turn on LED.
		1. Un-reset APB/GPIO interface
		2. Set LED pin (GPIO25) to SIO mode
		3. Set LED pin high (turn on)
	*/

	enable_iobank();

	set_led(false);

	// End state is infinite loop
	let mut i: usize;
	loop {
		// Turn LED OFF
		set_led(true);

		// Delay
		i = 0;
		while i < 0x50000 {
			nop();
			i += 1;
		}

		// Turn LED ON
		set_led(false);

		// Delay
		i = 0;
		while i < 0x50000 {
			nop();
			i += 1;
		}
	}
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
