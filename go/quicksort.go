package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func quick(dat []int, start,end int) {
	if (end-start) <= 1 {
		return
	}
	end -= 1 // End is exclusive

	piv := dat[end]

	// Place all values smaller than pivot in lower partition (partition values)
	tort, hare := start, start
	for ;hare < end;hare++ {
		if dat[hare] < piv {
			dat[tort], dat[hare] = dat[hare], dat[tort] // Swap values

			tort++ // Increment the tort
		}
	}

	dat[tort], dat[end] = dat[end], dat[tort] // Swap pivot with middle position

	quick(dat, start, tort) // Slower with goroutines
	quick(dat, tort+1, end+1)
}

func checkSort(dat []int, end int) bool {
	for i := 1;i < end;i++ {
		if dat[i-1] > dat[i] {
			return false
		}
	}

	return true
}

func main() {
	rand.Seed(time.Now().UnixNano())

	var N int = 10

	if len(os.Args) > 1 {
		n,err := strconv.Atoi(os.Args[1])
		if err == nil {
			N = n // Only set on no error
		}
	}

	nums := make([]int, N)
	for i,_ := range nums {
		nums[i] = rand.Intn(4*N)
	}
	//fmt.Println(nums)

	fmt.Printf("Sorting %7d elements..\n", N)

	start := time.Now()
	quick(nums, 0, len(nums))
	elapsed := time.Since(start)
	fmt.Println("Done in", elapsed)

	fmt.Println("Verifying..")
	sorted := checkSort(nums, len(nums))

	if sorted {
		fmt.Println("Sorted!")
	} else {
		fmt.Println("UNSORTED!")
	}

	//fmt.Println(nums)
}
