package native

import (
	"image"
	"image/draw"
)

// NewImageForDet builds a native Image directly from an in-memory image.Image
// for the detection path. It fills the RGB pixel buffer without any re-encode,
// preserving the source raster's fidelity — the pipeline already holds a
// decoded image, so re-encoding it would only lose information.
func NewImageForDet(src image.Image) (*Image, error) {
	b := src.Bounds()
	rgba := image.NewRGBA(b)
	draw.Draw(rgba, b, src, b.Min, draw.Src)
	w, h := b.Dx(), b.Dy()
	pix := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := rgba.PixOffset(x, y)
			d := (y*w + x) * 3
			pix[d] = rgba.Pix[o]     // R
			pix[d+1] = rgba.Pix[o+1] // G
			pix[d+2] = rgba.Pix[o+2] // B
		}
	}
	return &Image{W: w, H: h, Pix: pix}, nil
}
