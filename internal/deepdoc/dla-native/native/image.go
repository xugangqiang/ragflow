package native

// image.go — format-agnostic image decoding shared by all recognizers.
//
// DeepDoc's models all consume BGR-ordered pixels (DLA/TSR reach it via a
// cv2.cvtColor(rgb, BGR2RGB) swap on a PIL RGB image; the OCR models are fed
// cv2-decoded BGR directly). Decoding therefore yields RGB and callers ask for
// BGR via ToBGR() — one consistent path, no per-task decode duplication.
//
// Decode uses image.Decode (format auto-detection) rather than a hard-coded
// jpeg.Decode so the comparison tool mirrors the Python service, which
// decodes whatever bytes it receives (PIL for DLA/TSR, cv2.imdecode for OCR)
// and is format-agnostic. The comparison tool must be able to load the same
// formats production sends — chiefly PNG, since the Go inference client
// transport now encodes pages/crops as lossless PNG.

import (
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"os"
)

// Image is a decoded raster in R,G,B byte order, row-major (len H*W*3).
type Image struct {
	W, H int
	Pix  []byte
	// Path is the source file, used by the gocv build to re-decode via
	// cv2 (gocv.IMRead) so the preprocessing blob matches deepdoc's
	// cv2/PIL decode exactly. Empty for in-memory images (tests).
	Path string
	// Bytes is an in-memory encoded source (JPEG), used by the gocv build to
	// re-decode via cv2 (gocv.IMDecode) without a temp file. Set by
	// NewImageForDet; empty otherwise. Exactly one of Path/Bytes is used by
	// the gocv preprocessor (Bytes preferred when present).
	Bytes []byte
}

// Decode reads an image file (any format Go's image package can decode,
// e.g. PNG/JPEG) and returns it as RGB pixels. Format is auto-detected,
// matching the Python service's format-agnostic decode.
func Decode(path string) (*Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	rgba := image.NewRGBA(b)
	draw.Draw(rgba, b, img, b.Min, draw.Src)
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

// ToBGR returns a copy of the pixels with R and B channels swapped (B,G,R
// order), which is the channel order every DeepDoc ONNX expects.
func (im *Image) ToBGR() []byte {
	bgr := make([]byte, len(im.Pix))
	for i := 0; i < len(im.Pix); i += 3 {
		bgr[i] = im.Pix[i+2]
		bgr[i+1] = im.Pix[i+1]
		bgr[i+2] = im.Pix[i]
	}
	return bgr
}
