use crate::consts::*;
use crate::registers::Register;

fn enable_iobank() {
	let resets = unsafe { Register::new(RESETS_BASE as *mut u32) };
	let resets_status = unsafe { Register::new(RESETS_RESET_DONE as *mut u32) };

	// (1.) Write to reset register to enable IO_BANK0 (for GPIO), then wait for ready
	resets.atomic_bitset(0x20);
	while resets_status.read() & 0x20 == 0x0 {}
}
