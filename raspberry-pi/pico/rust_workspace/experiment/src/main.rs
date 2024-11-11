#![no_std]
#![no_main]

use pico;

fn main() {
}

#[panic_handler]
fn panic( _info: &core::panic::PanicInfo) -> ! {
	loop {}
}

#[no_mangle]
pub extern "C" fn _strat() -> ! {
	_ = main();

	loop {}
}
