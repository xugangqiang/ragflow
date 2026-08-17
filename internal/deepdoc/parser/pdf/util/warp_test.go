package util

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// makeWarpTestImage returns a w×h RGBA filled with white plus a dark diagonal
// stripe, so a warped crop differs from its surroundings and pixels can be
// compared against FastCrop for the axis-aligned case.
func makeWarpTestImage(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.White)
		}
	}
	for i := 0; i < w && i < h; i++ {
		img.Set(i, i, color.Black)
	}
	return img
}

// TestWarpCrop_AxisAlignedEqualsFastCrop proves the axis-aligned fast path is
// bit-identical to FastCrop: WarpCrop must not resample an upright quad.
func TestWarpCrop_AxisAlignedEqualsFastCrop(t *testing.T) {
	src := makeWarpTestImage(100, 60)
	pts := [4]Pt{{10, 10}, {90, 10}, {90, 50}, {10, 50}}
	got := WarpCrop(src, pts)
	want := FastCrop(src, 10, 10, 90, 50)
	if got.Bounds() != want.Bounds() {
		t.Fatalf("axis-aligned WarpCrop bounds %v != FastCrop bounds %v", got.Bounds(), want.Bounds())
	}
	for y := 0; y < got.Bounds().Dy(); y++ {
		for x := 0; x < got.Bounds().Dx(); x++ {
			if got.RGBAAt(x, y) != want.RGBAAt(x, y) {
				t.Fatalf("axis-aligned pixel mismatch at %d,%d", x, y)
			}
		}
	}
}

// TestWarpCrop_SlantedProducesRectangularCrop proves a genuinely slanted quad
// de-skews into the expected (W,H) rectangle without panicking.
func TestWarpCrop_SlantedProducesRectangularCrop(t *testing.T) {
	src := makeWarpTestImage(200, 200)
	pts := [4]Pt{{20, 40}, {160, 20}, {180, 120}, {40, 140}}
	got := WarpCrop(src, pts)
	if got == nil {
		t.Fatal("nil crop for slanted quad")
	}
	if got.Bounds().Dx() <= 0 || got.Bounds().Dy() <= 0 {
		t.Fatalf("empty warped crop: %v", got.Bounds())
	}
	wWant := int(math.Max(dist(pts[0], pts[1]), dist(pts[2], pts[3])))
	hWant := int(math.Max(dist(pts[0], pts[3]), dist(pts[1], pts[2])))
	if got.Bounds().Dx() != wWant || got.Bounds().Dy() != hWant {
		t.Fatalf("warped size %v != expected %dx%d", got.Bounds(), wWant, hWant)
	}
}

// TestWarpCrop_DegenerateAndNonFiniteSafe proves the untrusted-detector
// guards (non-finite coords, collinear/degenerate quad) never panic and
// always return a usable image.
func TestWarpCrop_DegenerateAndNonFiniteSafe(t *testing.T) {
	src := makeWarpTestImage(50, 50)
	// Non-finite coordinate must not panic.
	_ = WarpCrop(src, [4]Pt{{0, 0}, {math.NaN(), 0}, {10, 10}, {0, 10}})
	// Collinear (degenerate) quad must fall back, not panic.
	collinear := [4]Pt{{0, 0}, {10, 0}, {20, 0}, {30, 0}}
	if got := WarpCrop(src, collinear); got == nil {
		t.Fatal("nil crop on degenerate quad")
	}
	// Out-of-range quad must clamp, not panic / OOM.
	outOfRange := [4]Pt{{-100, -100}, {1000, -50}, {1200, 1200}, {-50, 1000}}
	if got := WarpCrop(src, outOfRange); got == nil {
		t.Fatal("nil crop on out-of-range quad")
	}
}
