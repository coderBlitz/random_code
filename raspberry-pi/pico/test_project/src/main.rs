//! Pico compilation and flashing test
//!
//! Simple file to test compilation of rust code for flashing the pi pico.

#![no_std]
#![no_main]

#![allow(dead_code)]

use core::arch::asm;
use core::panic::PanicInfo;

const RESETS_BASE: u32 = 0x4000_C000;
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
/// TODO: Try to interact with GPIO LED
#[no_mangle]
pub extern "C" fn _strat() -> ! {
	/* Attempt to turn on LED.
		1. Un-reset APB/GPIO interface
		2. Set LED pin (GPIO25) to SIO mode
		3. Set LED pin high (turn on)
	*/

	// Make pointers for registers needed
	let resets = RESETS_BASE as *mut u32;
	let resets_status = RESETS_RESET_DONE as *mut u32;

	// (1.) Write to reset register to enable IO_BANK0 (for GPIO), then wait for ready
	let reset_state = 0xFFFF_FFDF;
	unsafe { resets.write_volatile(reset_state); }
	while unsafe { resets_status.read_volatile() & 0x20 } == 0x0 {
	}

	// (2., 3.) Set pin mode
	let pin25_ctrl = GPIO25_CTRL as *mut u32;
	const LED_PIN_STATE_ON: u32 = 0x0000_3305;
	const LED_PIN_STATE_OFF: u32 = 0x0000_3205;
	unsafe { pin25_ctrl.write_volatile(LED_PIN_STATE_ON); }

	// End state is infinite loop
	let mut i;
	loop {
		// Turn LED OFF
		unsafe { pin25_ctrl.write_volatile(LED_PIN_STATE_OFF); }

		// Delay
		i = 0;
		while i < 0x50000 {
			nop();
			i += 1;
		}

		// Turn LED ON
		unsafe { pin25_ctrl.write_volatile(LED_PIN_STATE_ON); }

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
		asm!("NOP", options(nomem, nostack, raw));
	}
}

#[panic_handler]
fn panic( _info: &PanicInfo) -> ! {
	loop {}
}
