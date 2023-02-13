/*
For all unique prime factors, divisors are all combinations of factors, i.e.
 the power set of factors. Power set size is 2^n. So need first triangle with
 power set size > 500.
If all prime factors are unique, then 9 prime factors are needed for > 500 divisors.
If prime factors are repeated, then
The smallest number with 9 unique prime factors is primorial(9), is 223092870. This should
 be a lower bound for out answer.

First triangle above this is tri(21123), or 223101126

AFTER SOLVING:
Above is wrong. Solution is much smaller than 223101126, at ~75 million.
*/

fn triangle(n: usize) -> usize {
	(n * n + n) / 2
}

fn count_divisors(n: usize) -> usize {
	let max = f64::floor(f64::sqrt(n as f64)) as usize;
	let mut count = 0;

	for i in 1..max {
		if n % i == 0 {
			count += 1;
		}
	}

	count
}

fn main() {
	//let min = 21123;
	let mut k = 1;

	//for k in min..2*min {
	loop {
		let res = count_divisors(triangle(k));
		if res > 249 {
			println!("tri({k}) has {} divisors", res);
			println!("tri({k}) = {}", triangle(k));
			break;
		}

		k += 1;
	}
}
