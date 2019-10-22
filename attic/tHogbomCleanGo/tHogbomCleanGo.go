package main

import "fmt"
import "math"
import "os"
import "log"
import "time"
import "encoding/binary"

const (
	// Parameters
	dirtyFile         = "dirty.img"
	psfFile           = "psf.img"
	niters    uint32  = 1000
	gain      float32 = 0.1
	threshold float32 = 0.00001
)

func readImage(filename string) []float32 {
	f, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}

	// Read the image into the array
	fi, err := f.Stat()
	if err != nil {
		log.Fatal(err)
	}

	img := make([]float32, fi.Size()/4)
	rerr := binary.Read(f, binary.LittleEndian, &img)
	if rerr != nil {
		log.Fatal(rerr)
	}
	f.Close()
	return img
}

func writeImage(filename string, img []float32) {
	f, err := os.Create(filename)
	if err != nil {
		log.Fatal(err)
	}
	werr := binary.Write(f, binary.LittleEndian, &img)
	if werr != nil {
		log.Fatal(werr)
	}

	f.Close()
}

func checkSquare(image []float32) uint64 {
	var size = len(image)
	var singleDim = math.Sqrt(float64(size))
	if int(singleDim*singleDim) != size {
		log.Fatal("Error: Image is not square")
	}

	return uint64(singleDim)
}

func main() {
	// Load dirty image and psf
	fmt.Println("Reading dirty image and psf image")
	var dirty []float32 = readImage(dirtyFile)
	var dim = checkSquare(dirty)
	var psf []float32 = readImage(psfFile)
	var psfDim = checkSquare(psf)

	// Reports some numbers
	fmt.Println("Iterations = ", niters)
	fmt.Println("Image dimensions = ", dim, "x", dim)

	// Now we can do the timing for the CPU implementation
	fmt.Println("+++++ Cleaning (CPU) +++++")
	start := time.Now()
	model, residual := deconvolve(dirty, dim, psf, psfDim, niters, gain, threshold)
	time := time.Now().Sub(start).Seconds()
	restored := restore(model, residual)

	// Report on timings
	fmt.Println("    Time ", time, " (s) ")
	fmt.Println("    Time per cycle ", float32(time)/float32(niters)*1000.0, " (ms)")
	fmt.Println("    Cleaning rate  ", float32(niters)/float32(time), " (iterations per second)")
	fmt.Println("Done")

	// Write images out
	writeImage("residual.img", residual)
	writeImage("model.img", model)
	writeImage("restored.img", restored)
}
