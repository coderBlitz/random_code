package main

import (
	//"encoding/json"
	"fmt"
	"io"
	"os"
)

func main() {
	argc := len(os.Args)

	if argc != 2 {
		fmt.Println("Usage: ./json file.json")
		os.Exit(1)
	}

	fname := os.Args[1]
	fmt.Printf("Opening '%s'\n", fname)

	src, err := os.Open(fname)
	if err != nil {
		println(err.Error())
		os.Exit(2)
	}

	dat := make([]byte, 16) // Data array

	// Loop and print entire file
	err = nil
	var ret int
	for err == nil {
		ret, err = src.Read(dat)
		fmt.Print(string(dat[:ret]))
	}
	src.Close()

	// Gap line and any possible error
	fmt.Println()
	if err != io.EOF {
		println(err.Error())
		err = nil
	}
}
