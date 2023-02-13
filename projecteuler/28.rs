/*
Worked out formula by hand (quadratic equations)

For diagonals of length k, the sum of the diagonals is:
16/6*(2*k^3 - 3*k^2 + k) + 2*k*(k + 1) - 3
*/

// NOTE: The division by 3 *must* be last, since the result is guaranteed an integer (int division might round too early otherwise)
fn diag_sum(k: isize) -> isize {
	8 * k * (k * (2*k - 3) + 1) / 3 + 2*k*(k+1) - 3
}

fn main() {
	for i in 1..=4 {
		let sum = diag_sum(i);

		println!("Sum for {i} diags = {sum}");
	}

	let sum = diag_sum(501);

	println!("Sum for {} diags = {sum}", 1001);
}
