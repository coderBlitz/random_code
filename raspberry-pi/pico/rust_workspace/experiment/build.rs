fn main() {
	println!("cargo::rerun-if-changed=../flashmap.ld");
	println!("cargo::rustc-link-arg-bins=-T");
	println!("cargo::rustc-link-arg-bins=flashmap.ld");
}
