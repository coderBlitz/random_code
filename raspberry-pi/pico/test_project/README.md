# Experimental Rust project for Pi Pico
My simultaneous foray into embedded programming and Rust embedded programming.

# Instructions
* `cargo build`
* `arm-none-eabi-objcopy -O binary -j .text path/to/rust/binary out.bin`
* `./uf2 path/to/rust/binary`
* Flash the resulting UF2 to pi
* Profit
