fn main() {
	let mut sum = 2;
	let mut a = 0;
	let mut b = 2;
	let mut c = 0;
	let n = 4000000;

	while c < n {
		sum += c;

		c = 4*b + a;
		a = b;
		b = c;
	}

	println!("Sum evens = {sum}");
}
