package main

import "fmt"
import "math"
import "log"

func xpos(idx uint64, width uint64) uint64 {
	return idx % width
}

func ypos(idx uint64, width uint64) uint64 {
	return idx / width
}

func posToIdx(width uint64, x uint64, y uint64) uint64 {
	return (y * width) + x
}

func findPeak(image []float32) (maxVal float32, maxPos uint64) {
	maxVal = 0.0
	maxPos = 0
	for i := range image {
		absVal := float32(math.Abs(float64(image[i])))
		if absVal > maxVal {
			maxVal = absVal
			maxPos = uint64(i)
		}
	}
	return
}

func subtractPSF(psf []float32,
	psfWidth uint64,
	residual []float32,
	residualWidth uint64,
	peakPos uint64, psfPeakPos uint64,
	absPeakVal float32,
	gain float32) {
	var rx = float64(xpos(peakPos, residualWidth))
	var ry = float64(ypos(peakPos, residualWidth))

	var px = float64(xpos(psfPeakPos, psfWidth))
	var py = float64(ypos(psfPeakPos, psfWidth))

	var diffx uint64 = uint64(rx - px)
	var diffy uint64 = uint64(ry - py)

	var startx = math.Max(0, float64(rx - px))
	var starty = math.Max(0, float64(ry - py))

	var stopx = math.Min(float64(residualWidth-1), rx+(float64(psfWidth)-px-1))
	var stopy = math.Min(float64(residualWidth-1), ry+(float64(psfWidth)-py-1))

	factor := gain * absPeakVal
	for y := uint64(starty); y <= uint64(stopy); y++ {
		for x := uint64(startx); x <= uint64(stopx); x++ {
			residual[posToIdx(residualWidth, x, y)] -= factor *
				psf[posToIdx(psfWidth, x-diffx, y-diffy)]
		}
	}
}

func deconvolve(dirty []float32,
	dirtyWidth uint64,
	psf []float32,
	psfWidth uint64,
	niters uint32,
	gain float32,
	threshold float32) (model []float32, residual []float32) {
	// Make the model and residual
	model = make([]float32, len(dirty))
	for i := range model {
		model[i] = 0.0
	}
	residual = make([]float32, len(dirty))
	copy(residual, dirty)

	psfPeakVal, psfPeakPos := findPeak(psf)
	fmt.Println("Found peak of PSF: Maximum = ", psfPeakVal, " at location ",
		xpos(psfPeakPos, psfWidth), ",", ypos(psfPeakPos, psfWidth))

	for i := 0; i < int(niters); i++ {
		absPeakVal, absPeakPos := findPeak(residual)
		fmt.Println("Iteration: ", i+1, " - Maximum = ", absPeakVal, " at location ",
			xpos(absPeakPos, dirtyWidth), ",", ypos(absPeakPos, dirtyWidth))

		// Check if threshold has been reached
		if math.Abs(float64(absPeakVal)) < float64(threshold) {
			fmt.Println("Reached stopping threshold")
			break
		}

		// Add to model
		model[absPeakPos] += absPeakVal * gain

		// Subtract the PSF from the residual image
		subtractPSF(psf, psfWidth, residual, dirtyWidth, absPeakPos, psfPeakPos, absPeakVal, gain)
	}
	return
}

// TODO: This is not a real restore. Need to fit the primary beam and
// add a clean beam to the restored image, not just the raw components
func restore(model []float32, residual []float32) []float32 {
	if len(model) != len(residual) {
		log.Fatal("Error: residual and model have unequal size")
	}

	restored := make([]float32, len(residual))
	for i := range restored {
		restored[i] += model[i] + residual[i]
	}
	return restored
}
