//! Radius of circle from chord and partial diameter/radius.
//!
//! Equation is `R = (L^2 + 4D^2) / 8D`, where L is chord length and D is a
//!  partial radius. Partial radius is the distance between the arc and the
//!  chord, essentially splitting the full radius into two segments of lengths
//!  (R-D) and D.

fn main() {
	let chord_length: f64 = 1.;
	let partial_radius: f64 = 50.;

	let radius = (chord_length.powi(2) + 4. * partial_radius.powi(2)) / (8. * partial_radius);

	println!("Chord len = {chord_length}");
	println!("Partial radius = {partial_radius}");
	println!("==> Radius = {radius}");
}
