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
	fmt.Printf("\x1b[38;5;197mAnigo\x1b[0m  Animated image generator\n")
	fmt.Printf("Copyright (c) 2017 by White Wolf Studio\n\n")

	filenamePtr := flag.String("name", fmt.Sprintf("%d", time.Now().UnixNano()), "name of an exported image file")
	widthPtr := flag.Int("width", 500, "width of an exported image file")
	heightPtr := flag.Int("height", 500, "height of an exported image file")
	sharpnessPtr := flag.Float64("sharpness", 0.07, "sharpness of the image")
	focusPtr := flag.Float64("focus", 0.7, "focus to the center of the image")
	seedPtr := flag.Int64("seed", 0, "seed for random generation")

	flag.Parse()

	filename := *filenamePtr
	width, height := *widthPtr, *heightPtr
	sharpness := *sharpnessPtr
	focus := *focusPtr
	seed := *seedPtr

	fmt.Printf("------------+------------------------\n")
	fmt.Printf("File name   | %s\n", filename+".gif")
	fmt.Printf("Image size  | (%d x %d)\n", width, height)
	fmt.Printf("Sharpness   | %f\n", sharpness)
	fmt.Printf("Focus       | %f\n", focus)
	fmt.Printf("Seed        | %d\n", seed)
	fmt.Printf("------------+------------------------\n")

	rand.Seed(seed)

	img := &gif.GIF{}
	cppn := NewCPPN(&CPPNParam{5, 8, 16, 3})

	fmt.Println("\x1b[?25l")
	fmt.Printf("Processing... [")
	for i := 0; i < 10; i++ {
		emoji.Print(":fish:")
	}
	fmt.Printf("]\x1b[21D")

	for theta := 0; theta < 360; theta += 6 {
		frame := image.NewPaletted(image.Rect(0, 0, width, height), palette.Plan9)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				// process inputs
				xs := float64(x) * sharpness
				ys := float64(y) * sharpness
				r := math.Sqrt(math.Pow(float64(x-width/2), 2.0)+
					math.Pow(float64(y-height/2), 2.0)) * sharpness * focus
				z1 := math.Sin(float64(theta)) * 0.5
				z2 := math.Cos(float64(theta)) * 0.5

				inputs := []float64{xs, ys, r, z1, z2}
				output := cppn.FeedForward(inputs)
				frame.Set(x, y, color.RGBA{
					uint8(255 * output[0]),
					uint8(255 * output[1]),
					uint8(255 * output[2]),
					255,
				})
			}
		}
		img.Image = append(img.Image, frame)
		img.Delay = append(img.Delay, 0)

		if theta%36 == 0 {
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
