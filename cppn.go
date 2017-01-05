package main

import "fmt"

type Layer struct {
	weights
	activation *Activation
}

type CPPN struct {
	layers []*Layer
}

func main() {
	fmt.Println("vim-go")
}
