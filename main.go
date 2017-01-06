/*


anigo  Animated image generation with random CPPN

@licstart   The following is the entire license notice for
the Go code in this page.

Copyright (C) 2017 jin yeom, whitewolf.studio

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

As additional permission under GNU GPL version 3 section 7, you
may distribute non-source (e.g., minimized or compacted) forms of
that code without the copy of the GNU GPL normally required by
section 4, provided you include this license notice and a URL
through which recipients can access the Corresponding Source.

@licend    The above is the entire license notice
for the Go code in this page.


*/

package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/gif"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/gonum/matrix/mat64"
	"github.com/kyokomi/emoji"
)

type Activation struct {
	Name string
	Fn   func(x float64) float64
}

func Sigmoid() *Activation {
	return &Activation{
		Name: "sigmoid",
		Fn: func(x float64) float64 {
			return 1.0 / (1.0 + math.Exp(-x))
		},
	}
}

func Tanh() *Activation {
	return &Activation{
		Name: "tanh",
		Fn:   math.Tanh,
	}
}

type Layer struct {
	Weights    *mat64.Dense // connection weights from previous layer
	Activation *Activation  // activation function for feed forwarding
}

func NewLayer(r, c int, activation *Activation) *Layer {
	weights := make([]float64, r*c)
	for i := range weights {
		weights[i] = rand.NormFloat64() * 0.5
	}

	return &Layer{
		Weights:    mat64.NewDense(r, c, weights),
		Activation: activation,
	}
}

func (l *Layer) Activate(signal *mat64.Vector) *mat64.Vector {
	withBias := mat64.NewVector(signal.Len()+1, nil)
	withBias.CopyVec(signal)
	withBias.SetVec(signal.Len(), -1.0)

	_, c := l.Weights.Dims()
	activated := mat64.NewVector(c, make([]float64, c))
	activated.MulVec(l.Weights.T(), withBias)
	for i := 0; i < c; i++ {
		activated.SetVec(i, l.Activation.Fn(activated.At(i, 0)))
	}

	return activated
}

type CPPNParam struct {
	NumInputs        int // number of inputs
	NumHiddenLayers  int // number of hidden layers
	NumHiddenNeurons int // number of neurons in a hidden layer
	NumOutputs       int // number of outputs
}

type CPPN struct {
	Param  *CPPNParam
	Layers []*Layer
}

func NewCPPN(param *CPPNParam) *CPPN {
	return &CPPN{
		Param: param,
		Layers: func() []*Layer {
			layers := make([]*Layer, 0, param.NumHiddenLayers+1)

			layers = append(layers, NewLayer(
				param.NumInputs+1,
				param.NumHiddenNeurons,
				Tanh()))

			for i := 1; i < param.NumHiddenLayers; i++ {
				layers = append(layers, NewLayer(
					param.NumHiddenNeurons+1,
					param.NumHiddenNeurons,
					Tanh()))
			}

			layers = append(layers, NewLayer(
				param.NumHiddenNeurons+1,
				param.NumOutputs,
				Sigmoid()))

			return layers
		}(),
	}
}

func (c *CPPN) FeedForward(inputs []float64) []float64 {
	if len(inputs) != c.Param.NumInputs {
		panic(fmt.Errorf("invalid number of inputs: %d\n", len(inputs)))
	}

	signalVec := mat64.NewVector(len(inputs), inputs)
	for _, layer := range c.Layers {
		outputs := layer.Activate(signalVec)
		signalVec.CloneVec(outputs)
	}

	return signalVec.RawVector().Data
}

func main() {
	fmt.Printf("\x1b[38;5;160mAnigo\x1b[0m  Animated image generator\n")
	fmt.Printf("Copyright (c) 2017 by White Wolf Studio\n")

	filenamePtr := flag.String("name", fmt.Sprintf("%d", time.Now().UnixNano()), "name of an exported image file")
	widthPtr := flag.Int("width", 200, "width of an exported image file")
	heightPtr := flag.Int("height", 200, "height of an exported image file")
	sharpnessPtr := flag.Float64("sharpness", 0.07, "sharpness of the image")
	focusPtr := flag.Float64("focus", 1.0, "focus to the center of the image")
	seedPtr := flag.Int64("seed", 0, "seed for random generation")
	depthPtr := flag.Int("depth", 12, "number of hidden layers")
	sizePtr := flag.Int("size", 24, "number of neurons in a hidden layer")
	patternPtr := flag.Bool("pattern", false, "true if exporting a patterned image")
	densityPtr := flag.Float64("density", 1.0, "density of patterns (valid only with -pattern flag)")
	grayPtr := flag.Bool("gray", false, "true if exporting a black and white image")

	flag.Parse()

	filename := *filenamePtr
	width, height := *widthPtr, *heightPtr
	sharpness := *sharpnessPtr
	focus := *focusPtr
	seed := *seedPtr
	depth := *depthPtr
	size := *sizePtr
	pattern := *patternPtr
	density := *densityPtr
	gray := *grayPtr

	fmt.Printf("------------+-----------------------\n")
	fmt.Printf("File name   | %s\n", filename+".gif")
	fmt.Printf("Image size  | (%d x %d)\n", width, height)
	fmt.Printf("Sharpness   | %f\n", sharpness)
	fmt.Printf("Focus       | %f\n", focus)
	fmt.Printf("Seed        | %d\n", seed)
	fmt.Printf("Depth       | %d\n", depth)
	fmt.Printf("Size        | %d\n", size)
	fmt.Printf("Pattern     | %t\n", pattern)
	if pattern {
		fmt.Printf("Density     | %f\n", density)
	}
	fmt.Printf("Gray        | %t\n", gray)
	fmt.Printf("------------+-----------------------\n")

	rand.Seed(seed)

	img := &gif.GIF{}

	var param *CPPNParam
	var colors []color.Color

	if gray {
		param = &CPPNParam{5, depth, size, 1}
		colors = make([]color.Color, 0, 256)
		for i := 0; i < 255; i++ {
			colors = append(colors, color.Gray{uint8(i)})
		}
	} else {
		param = &CPPNParam{5, depth, size, 3}
		colors = palette.Plan9
	}

	cppn := NewCPPN(param)

	fmt.Printf("\x1b[?25l")
	fmt.Printf("\x1b[1mProcessing... [")
	for i := 0; i < 10; i++ {
		emoji.Print(":fish:")
	}
	fmt.Printf("]\x1b[0m\x1b[21D")

	drawPixel := func(pattern, gray bool) func(frame *image.Paletted, x, y, theta int) {
		var set func(*image.Paletted, []float64, int, int)
		if gray {
			set = func(frame *image.Paletted, output []float64, x, y int) {
				frame.Set(x, y, color.Gray{
					uint8(255 * output[0]),
				})
			}
		} else {
			set = func(frame *image.Paletted, output []float64, x, y int) {
				frame.Set(x, y, color.RGBA{
					uint8(255 * output[0]),
					uint8(255 * output[1]),
					uint8(255 * output[2]),
					255,
				})
			}
		}

		if pattern {
			return func(frame *image.Paletted, x, y, theta int) {
				xs := math.Sin(float64(x)*density) * sharpness
				ys := math.Cos(float64(y)*density) * sharpness
				r := math.Sqrt(math.Pow(float64(x-width/2), 2.0)+
					math.Pow(float64(y-height/2), 2.0)) * sharpness * focus
				z1 := math.Cos(float64(theta) / (math.Pi * 3.0))
				z2 := math.Sin(float64(theta) / (math.Pi * 3.0))

				inputs := []float64{xs, ys, r, z1, z2}
				output := cppn.FeedForward(inputs)
				set(frame, output, x, y)
			}
		}
		return func(frame *image.Paletted, x, y, theta int) {
			xs := float64(x) * sharpness
			ys := float64(y) * sharpness
			r := math.Sqrt(math.Pow(float64(x-width/2), 2.0)+
				math.Pow(float64(y-height/2), 2.0)) * sharpness * focus
			z1 := math.Cos(float64(theta) / (math.Pi * 3.0))
			z2 := math.Sin(float64(theta) / (math.Pi * 3.0))

			inputs := []float64{xs, ys, r, z1, z2}
			output := cppn.FeedForward(inputs)
			set(frame, output, x, y)
		}
	}(pattern, gray)

	runtime.GOMAXPROCS(4)
	var wg sync.WaitGroup

	drawPart := func(frame *image.Paletted, x0, x1, y0, y1, theta int) {
		defer wg.Done()
		for y := y0; y < y1; y++ {
			for x := x0; x < x1; x++ {
				drawPixel(frame, x, y, theta)
			}
		}
	}

	for theta := 0; theta < 60; theta++ {
		frame := image.NewPaletted(image.Rect(0, 0, width, height), colors)

		wg.Add(1)
		go drawPart(frame, 0, width/2, 0, height/2, theta)

		wg.Add(1)
		go drawPart(frame, width/2, width, 0, height/2, theta)

		wg.Add(1)
		go drawPart(frame, 0, width/2, height/2, height, theta)

		wg.Add(1)
		go drawPart(frame, width/2, width, height/2, height, theta)

		wg.Wait()

		img.Image = append(img.Image, frame)
		img.Delay = append(img.Delay, 0)

		if theta%6 == 5 {
			emoji.Printf(":sushi:")
		}
	}

	fmt.Printf("\x1b[2C")
	emoji.Println(":beer:")
	fmt.Printf("\x1b[?25h")

	f, err := os.OpenFile(filename+".gif", os.O_WRONLY|os.O_CREATE, 0600)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	gif.EncodeAll(f, img)
}
