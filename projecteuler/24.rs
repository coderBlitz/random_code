/*
Problem 24: Find millionth lexicographic permutation of the decimal digits (0-9).

Total permutations is 10!
The first 9! permutations begin with 0, the second 9! begin with 1, etc.
Use this logic to iteratively subtract the k'th factorial till one mil is reached.

Can easily verify that the millionth permutation will start with '2' (the third
 digit), since 2*9! < 1000000 < 3*9!
So the remaining digits are (0,1,3,4,5,6,7,8,9)
*/

fn factorial(n: usize) -> usize {
	let mut ret = 1;
	for i in 2..=n {
		ret *= i;
	}

	ret
}

fn main() {
	const N: usize = 1000000;

	// 0-indexed remainder
	let mut remainder: usize = N-1;

	let mut digits: Vec<usize> = (0..=9).collect();
	println!("digits = {digits:?}");

	let mut k = digits.len() - 1;
	let mut result = Vec::new();
	while remainder > 0 {
		let pos = remainder / factorial(k);

		let d = digits.remove(pos);
		result.push(d);

		remainder -= pos * factorial(k);
		k -= 1;
		println!("Remainder: {remainder}");
	}

	// Append remaining digits (front-to-back, increasing order)
	result.extend_from_slice(&digits);

	let s: String = result.into_iter().map(|x| x.to_string()).collect();
	println!("Result = {}", s);
} 
