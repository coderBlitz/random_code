package main

import (
	"fmt"
	"math"
	"math/cmplx"
)

func main() {
	v := 1 - 2i
	fmt.Println("Val =", v)

	mag := cmplx.Abs(v)

	b := v + complex(mag, 0) // New complex value is this sum
	b_mag := cmplx.Abs(b) // Normalize b
	b /= complex(b_mag, 0)
	fmt.Println("B =", b)

	root := complex(math.Sqrt(mag), 0) * b // Scale b by square root of the magnitude
	fmt.Println("Root  =", root)
	fmt.Println("Check =", cmplx.Sqrt(v))
}
