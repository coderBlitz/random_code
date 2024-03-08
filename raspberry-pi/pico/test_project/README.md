# Experimental Rust project for Pi Pico
My simultaneous foray into embedded programming and Rust embedded programming.

# Instructions
* `cargo build`
* `arm-none-eabi-objcopy -O binary -j .text path/to/rust/binary out.bin` (probably don't want .text exclusively)
* `./uf2 path/to/rust/binary`
* Flash the resulting UF2 to pi
* Profit

## Stage 2 and beyond
Until better method implemented:
1. Compile binary with exclusively the `enable_xip()` function call.
2. Create UF2 of binary at offset 0x1000_0000.
3. Load binary onto pico.
4. Remove `enable_xip()` call and implement binary as desired.
5. Create UF2 of new binary at offset 0x1000_0100.

# Second stage bootloader steps
See code (for now).
