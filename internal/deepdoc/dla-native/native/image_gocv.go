//go:build gocv

package native

import (
	"bytes"
	"image"
	"image/jpeg"
)

// NewImageForDet builds a native Image for the gocv detection path. The gocv
// preprocessor re-decodes the source through cv2 (gocv.IMDecode) to match
// deepdoc's cv2/JPEG decode exactly, so we serialize the in-memory image to an
// in-memory JPEG buffer here instead of writing a temp file. The bytes are
// equivalent to what the old CreateTemp + jpeg.Encode produced, so cv2 decode
// parity is unchanged — only the disk round-trip is eliminated. See
// HANDOFF.md §8 A2.
func NewImageForDet(src image.Image) (*Image, error) {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, src, nil); err != nil {
		return nil, err
	}
	return &Image{
		W:     src.Bounds().Dx(),
		H:     src.Bounds().Dy(),
		Bytes: buf.Bytes(),
	}, nil
}
