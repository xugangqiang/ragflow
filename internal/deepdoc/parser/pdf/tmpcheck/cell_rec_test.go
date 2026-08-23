//go:build native_det && integration

package tmpcheck

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"native"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	pdfpkg "ragflow/internal/deepdoc/parser/pdf"
	pdftype "ragflow/internal/deepdoc/parser/pdf/type"
	infnative "ragflow/internal/deepdoc/parser/pdf/inference/native_analyzer"
	inf "ragflow/internal/deepdoc/parser/pdf/inference"
	"ragflow/internal/deepdoc/parser/pdf/util"
)

const matchIoUThr = 0.5

type quad [4][2]float64

func boxToQuad(b pdftype.OCRBox) quad {
	return quad{{b.X0, b.Y0}, {b.X1, b.Y1}, {b.X2, b.Y2}, {b.X3, b.Y3}}
}

func quadIoU(a, b quad) float64 {
	// axis-aligned bbox IoU (detector boxes are ~axis-aligned)
	ax0 := math.Min(math.Min(a[0][0], a[1][0]), math.Min(a[2][0], a[3][0]))
	ay0 := math.Min(math.Min(a[0][1], a[1][1]), math.Min(a[2][1], a[3][1]))
	ax1 := math.Max(math.Max(a[0][0], a[1][0]), math.Max(a[2][0], a[3][0]))
	ay1 := math.Max(math.Max(a[0][1], a[1][1]), math.Max(a[2][1], a[3][1]))
	bx0 := math.Min(math.Min(b[0][0], b[1][0]), math.Min(b[2][0], b[3][0]))
	by0 := math.Min(math.Min(b[0][1], b[1][1]), math.Min(b[2][1], b[3][1]))
	bx1 := math.Max(math.Max(b[0][0], b[1][0]), math.Max(b[2][0], b[3][0]))
	by1 := math.Max(math.Max(b[0][1], b[1][1]), math.Max(b[2][1], b[3][1]))
	ix0 := math.Max(ax0, bx0)
	iy0 := math.Max(ay0, by0)
	ix1 := math.Min(ax1, bx1)
	iy1 := math.Min(ay1, by1)
	iw := ix1 - ix0
	ih := iy1 - iy0
	if iw <= 0 || ih <= 0 {
		return 0
	}
	i := iw * ih
	ua := (ax1 - ax0) * (ay1 - ay0)
	ub := (bx1 - bx0) * (by1 - by0)
	return i / (ua + ub - i)
}

type boxRec struct {
	i   int
	box pdftype.OCRBox
	q   quad
	text string // OCRRecognize joined
	cells int    // TSR cell count (0 if not table)
}

type cellDiff struct {
	Page      int     `json:"page"`
	IoU       float64 `json:"iou"`
	GoText    string  `json:"go_text"`
	PyText    string  `json:"py_text"`
	GoCells   int     `json:"go_cells"`
	PyCells   int     `json:"py_cells"`
	RecDiff   bool    `json:"rec_diff"`
	TSRDiff   bool    `json:"tsr_diff"`
}

type pageCellReport struct {
	PDF          string      `json:"pdf"`
	Page         int         `json:"page"`
	GoBoxes      int         `json:"go_boxes"`
	PyBoxes      int         `json:"py_boxes"`
	Matched      int         `json:"matched"`
	GoOnly       int         `json:"go_only"`
	PyOnly       int         `json:"py_only"`
	RecDiffs     int         `json:"rec_diffs"`
	TSRDiffs     int         `json:"tsr_diffs"`
	Diffs        []cellDiff  `json:"diffs"`
}

func cropBox(img image.Image, b pdftype.OCRBox) image.Image {
	// DeepDoc detection can emit degenerate boxes (zero width/height) at the
	// render DPI (e.g. a single text line rounds to Y0==Y1). CropImageRegion
	// rejects those, which would silently drop the box from rec/tsr. Expand
	// degenerate edges by a minimal pixel margin so the region is valid.
	const minSpan = 2.0
	x0, y0, x1, y1 := b.X0, b.Y0, b.X1, b.Y1
	if x1-x0 < minSpan {
		mid := (x0 + x1) / 2
		x0, x1 = mid-minSpan/2, mid+minSpan/2
	}
	if y1-y0 < minSpan {
		mid := (y0 + y1) / 2
		y0, y1 = mid-minSpan/2, mid+minSpan/2
	}
	r := pdftype.DLARegion{X0: x0, Y0: y0, X1: x1, Y1: y1}
	c, err := util.CropImageRegion(img, r)
	if err != nil {
		return nil
	}
	return c
}

func recText(tks []pdftype.OCRText) string {
	var sb strings.Builder
	for _, t := range tks {
		sb.WriteString(t.Text)
	}
	return sb.String()
}

func TestCellLevelCompare(t *testing.T) {
	pdfDir := os.Getenv("INPROC_PDF_DIR")
	if pdfDir == "" {
		pdfDir = filepath.Join("testdata", "real_pdfs")
	}
	outDir := os.Getenv("CELL_OUT")
	if outDir == "" {
		outDir = filepath.Join("testdata", "output", "render_compare", "cellrec")
	}
	os.MkdirAll(outDir, 0o755)
	modelDir := os.Getenv("MODEL_DIR")
	pyURL := os.Getenv("DEEPDOC_URL")
	if pyURL == "" {
		pyURL = "http://localhost:9390"
	}
	if err := native.InitORT(os.Getenv("ORT_LIB")); err != nil {
		t.Fatalf("InitORT: %v", err)
	}
	goAna, err := infnative.NewAnalyzer(modelDir, infnative.DefaultDropScore)
	if err != nil {
		t.Fatalf("goAna: %v", err)
	}
	pyAna, err := inf.NewClient(pyURL)
	if err != nil {
		t.Fatalf("pyAna: %v", err)
	}

	var names []string
	if sub := os.Getenv("INPROC_PDFS"); sub != "" {
		for _, s := range strings.Split(sub, ",") {
			s = strings.TrimSpace(s)
			if s != "" {
				names = append(names, s)
			}
		}
	} else {
		// default: first 35 PDFs (simple set), sorted for determinism
		entries, _ := os.ReadDir(pdfDir)
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".pdf") {
				continue
			}
			names = append(names, e.Name())
		}
		sort.Strings(names)
		if len(names) > 35 {
			names = names[:35]
		}
	}

	ctx := context.Background()
	totalRec, totalTSR := 0, 0
		for _, name := range names {
			func() {
				defer func() {
				if r := recover(); r != nil {
					t.Logf("%s: PANIC: %v", name, r)
				}
			}()
			data, err := os.ReadFile(filepath.Join(pdfDir, name))
			if err != nil {
				t.Logf("%s: read: %v", name, err)
				return
			}
			eng, err := pdfpkg.NewEngine(data)
			if err != nil {
				t.Logf("%s: engine: %v", name, err)
				return
			}
			defer eng.Close()
			pc, err := eng.PageCount()
			if err != nil {
				t.Logf("%s: pagecount: %v", name, err)
				return
			}
			pageCap := 0
			if v := os.Getenv("PAGE_CAP"); v != "" {
				fmt.Sscanf(v, "%d", &pageCap)
			}
			for p := 0; p < pc; p++ {
				if pageCap > 0 && p >= pageCap {
					break
				}
				img, err := pdfpkg.RenderPageToImage(eng, p)
				if err != nil {
					t.Logf("%s p%d render: %v", name, p, err)
					continue
				}
				goDet, err := goAna.OCRDetect(ctx, img)
				if err != nil {
					t.Logf("%s p%d det(go): %v", name, p, err)
					continue
				}
				pyDet, err := pyAna.OCRDetect(ctx, img)
				if err != nil {
					t.Logf("%s p%d det(py): %v", name, p, err)
					continue
				}
				// build quads
				gq := make([]quad, len(goDet))
				for i, b := range goDet {
					gq[i] = boxToQuad(b)
				}
				pq := make([]quad, len(pyDet))
				for i, b := range pyDet {
					pq[i] = boxToQuad(b)
				}
				// greedy match
				type cand struct {
					gi, pi int
					iou    float64
				}
				var cs []cand
				for i := range goDet {
					for j := range pyDet {
						if iou := quadIoU(gq[i], pq[j]); iou >= matchIoUThr {
							cs = append(cs, cand{i, j, iou})
						}
					}
				}
				sort.Slice(cs, func(i, j int) bool { return cs[i].iou > cs[j].iou })
				usedG := map[int]bool{}
				usedP := map[int]bool{}
				var matches []cand
				for _, c := range cs {
					if usedG[c.gi] || usedP[c.pi] {
						continue
					}
					usedG[c.gi], usedP[c.pi] = true, true
					matches = append(matches, c)
				}
				rep := pageCellReport{PDF: name, Page: p, GoBoxes: len(goDet), PyBoxes: len(pyDet), Matched: len(matches)}
				for i := range goDet {
					if !usedG[i] {
						rep.GoOnly++
					}
				}
				for i := range pyDet {
					if !usedP[i] {
						rep.PyOnly++
					}
				}
			for _, m := range matches {
				gb, pb := goDet[m.gi], pyDet[m.pi]
				gcrop := cropBox(img, gb)
				pcrop := cropBox(img, pb)
				if gcrop == nil || pcrop == nil {
					continue
				}
				// Skip boxes too small for meaningful rec/tsr. Degenerate
				// detector boxes (zero-height lines) expand to a 2px sliver
				// that is unreadable and can crash the native ORT rec path.
				if gcrop.Bounds().Dx() < 8 || gcrop.Bounds().Dy() < 8 ||
					pcrop.Bounds().Dx() < 8 || pcrop.Bounds().Dy() < 8 {
					continue
				}
				grec, gerr := goAna.OCRRecognize(ctx, gcrop)
				prec, perr := pyAna.OCRRecognize(ctx, pcrop)
				if perr != nil {
					t.Logf("%s p%d py_rec err (box %d): %v", name, p, m.pi, perr)
				}
				gt := recText(grec)
				pt := recText(prec)
				gtsr, gtsrErr := goAna.TSR(ctx, gcrop)
				ptsr, ptsrErr := pyAna.TSR(ctx, pcrop)
				if ptsrErr != nil {
					t.Logf("%s p%d py_tsr err (box %d): %v", name, p, m.pi, ptsrErr)
				}
				_ = gerr
				_ = gtsrErr
				recDiff := normText(gt) != normText(pt)
					tsrDiff := len(gtsr) != len(ptsr)
					if recDiff || tsrDiff {
						rep.Diffs = append(rep.Diffs, cellDiff{
							Page: p, IoU: m.iou, GoText: gt, PyText: pt,
							GoCells: len(gtsr), PyCells: len(ptsr),
							RecDiff: recDiff, TSRDiff: tsrDiff,
						})
						if recDiff {
							rep.RecDiffs++
							totalRec++
						}
						if tsrDiff {
							rep.TSRDiffs++
							totalTSR++
						}
					}
				}
				if rep.Diffs != nil || rep.GoOnly > 0 || rep.PyOnly > 0 {
					jb, _ := json.MarshalIndent(rep, "", "  ")
					_ = os.WriteFile(filepath.Join(outDir, fmt.Sprintf("%s_p%d.json", baseName(name), p)), jb, 0o644)
				}
			}
			fmt.Printf("%s: pages=%d done\n", name, pc)
		}()
	}
	fmt.Printf("DONE. rec_diffs=%d tsr_diffs=%d\n", totalRec, totalTSR)
}

func mustRec(ctx context.Context, a pdftype.DocAnalyzer, img image.Image) []pdftype.OCRText {
	t, err := a.OCRRecognize(ctx, img)
	if err != nil {
		return nil
	}
	return t
}

func normText(s string) string {
	var b strings.Builder
	for _, r := range s {
		if (r >= '0' && r <= '9') || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= 0x4e00 && r <= 0x9fff) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func baseName(name string) string {
	return strings.TrimSuffix(name, ".pdf")
}
