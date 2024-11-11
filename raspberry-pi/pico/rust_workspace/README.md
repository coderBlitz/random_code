# Experimental Rust project for Pi Pico
My simultaneous foray into embedded programming and Rust embedded programming.

# Instructions
* `cargo build`
* `make_uf2.sh path/to/bin`
* Flash the resulting UF2 to pi
* Profit

# Second stage bootloader steps
See code (for now).

# Notes/TODOs
* Eventually define intrinsic overwrites with ROM functions (for float, etc.).
   See [rp2040-hal source][1] for details/examples, and [the GCC docs][2] for
   the function names. Remember "weak symbols".

[1]: https://github.com/rp-rs/rp-hal/blob/main/rp2040-hal/src/float/div.rs
[2]: https://gcc.gnu.org/onlinedocs/gccint/Soft-float-library-routines.html
