/*
To find amicable numbers, start by finding divisors of numbers in the range,
 sum them up for each, and store in an array. Potentially use 0 for sums outside
 the question range (10000).
Loop through sums and (for those in range) check if pairs equal each other, then
 add to grand sum (one or both; if both, set both to 0 to skip).

A once-through solution would probably just check if paired number is less than
 current number, check if amicable pair, then add both if so. 
*/

fn main() {
	const N: usize = 10000;

	// Make a heap array (a workaround till Rust has a cleaner way)
	/*let mut sums = Vec::<usize>::with_capacity(N);
	unsafe { sums.set_len(N); }
	let mut sums = sums.into_boxed_slice();*/

	//let mut sums = Vec::<usize>::with_capacity(N);
	//sums.push(0);
	let mut sums = [0;N];

	// Loop over range
	let mut total = 0;
	for i in 1..N {
		let mut sum = 1; // Always added

		// Factorization loop
		for j in 2..=f64::sqrt(i as f64).floor() as usize {
			let (q, r) = (i / j, i % j);

			// If divisible, add both divisors
			if r == 0 {
				sum += j + q;

				// Subtract extra divisor when number is square
				if j == q {
					sum -= q;
				}
			}
		}

		sums[i] = sum;
		//sums.push(sum);

		// Must be less to have
		if sum < i {
			if sums[sum] == i {
				total += sum + i;
			}
		}
	}

	println!("Total = {total}");
}
