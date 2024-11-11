
const ATOMIC_XOR: usize = 0x1000;
const ATOMIC_BITMASK_SET: usize = 0x2000;
const ATOMIC_BITMASK_CLEAR: usize = 0x3000;

/// Pico peripheral register.
///
pub struct Register(*mut u32);
impl Register {
	/// Construct a register from the given address.
	///
	/// # Safety
	/// The user must ensure the address is a valid register, as defined in the
	///  RP2040 datasheet.
	pub unsafe fn new(reg: *mut u32) -> Self {
		Register(reg)
	}

	pub fn write(&mut self, value: u32) {
		unsafe {
			self.0.write_volatile(value);
		}
	}
	pub fn read(&self) -> u32 {
		unsafe {
			self.0.read_volatile()
		}
	}

	/// Atomically XOR the register with the given value.
	pub fn atomic_xor(&self, value: u32) {
		unsafe {
			self.0.byte_add(ATOMIC_XOR).write_volatile(value)
		}
	}

	/// Atomically set bits in register given by `value`.
	pub fn atomic_bitset(&self, value: u32) {
		unsafe {
			self.0.byte_add(ATOMIC_BITMASK_SET).write_volatile(value)
		}
	}

	/// Atomically clear bits in register given by `value`.
	pub fn atomic_bitclear(&self, value: u32) {
		unsafe {
			self.0.byte_add(ATOMIC_BITMASK_CLEAR).write_volatile(value)
		}
	}
}
