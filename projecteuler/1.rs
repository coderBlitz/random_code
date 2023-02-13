#![allow(non_snake_case)]

fn triangle(n: usize) -> usize {
	(n * n + n) / 2
}

fn main() {
	let N = 999;
	let a = N / 3;
	let b = N / 5;
	let c = N / 15;

	let sum3 = 3 * triangle(a);
	let sum5 = 5 * triangle(b);
	let sum15 = 15 * triangle(c);

	let ans = (sum3 + sum5) - sum15;
	println!("Ans = {}", ans);
}
