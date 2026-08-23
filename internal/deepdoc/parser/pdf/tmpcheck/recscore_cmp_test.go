//go:build native_det && integration

package tmpcheck

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"native"
	pdfpkg "ragflow/internal/deepdoc/parser/pdf"
	infnative "ragflow/internal/deepdoc/parser/pdf/inference/native_analyzer"
	inf "ragflow/internal/deepdoc/parser/pdf/inference"
)

// TestRecScoreCompare measures, on the same matched detection boxes, the
// recognition confidence each backend assigns, to find hard evidence of a
// confidence-calibration divergence (the dominant cause of Go's ~5.8% lower
// line count at dropScore=0.5). It does NOT re-run the full corpus: scope it
// with INPROC_PDFS and PAGE_CAP.
func TestRecScoreCompare(t *testing.T) {
	pdfDir := os.Getenv("INPROC_PDF_DIR")
	if pdfDir == "" {
		pdfDir = filepath.Join("testdata", "real_pdfs")
	}
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
		entries, _ := os.ReadDir(pdfDir)
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".pdf") {
				continue
			}
			names = append(names, e.Name())
		}
		sort.Strings(names)
		if len(names) > 3 {
			names = names[:3]
		}
	}

	pageCap := 0
	if v := os.Getenv("PAGE_CAP"); v != "" {
		fmt.Sscanf(v, "%d", &pageCap)
	}

	ctx := context.Background()
	var n, goDropPyKeep, pyDropGoKeep, bothHigh int
	var sumGo, sumPy, goHighSum, pyHighSum float64
	type sample struct {
		PDF, GoText, PyText string
		GoConf, PyConf      float64
	}
	var samples []sample
	sampleCap := 12
	for _, name := range names {
		data, err := os.ReadFile(filepath.Join(pdfDir, name))
		if err != nil {
			t.Logf("%s: read: %v", name, err)
			continue
		}
		eng, err := pdfpkg.NewEngine(data)
		if err != nil {
			t.Logf("%s: engine: %v", name, err)
			continue
		}
		pc, err := eng.PageCount()
		if err != nil {
			eng.Close()
			continue
		}
		for p := 0; p < pc; p++ {
			if pageCap > 0 && p >= pageCap {
				break
			}
			img, err := pdfpkg.RenderPageToImage(eng, p)
			if err != nil {
				continue
			}
			goDet, err := goAna.OCRDetect(ctx, img)
			if err != nil {
				continue
			}
			pyDet, err := pyAna.OCRDetect(ctx, img)
			if err != nil {
				continue
			}
			gq := make([]quad, len(goDet))
			for i, b := range goDet {
				gq[i] = boxToQuad(b)
			}
			pq := make([]quad, len(pyDet))
			for i, b := range pyDet {
				pq[i] = boxToQuad(b)
			}
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
			for _, c := range cs {
				if usedG[c.gi] || usedP[c.pi] {
					continue
				}
				usedG[c.gi], usedP[c.pi] = true, true
				gc := cropBox(img, goDet[c.gi])
				pc2 := cropBox(img, pyDet[c.pi])
				if gc == nil || pc2 == nil {
					continue
				}
				grec, gerr := goAna.OCRRecognize(ctx, gc)
				prec, perr := pyAna.OCRRecognize(ctx, pc2)
				if gerr != nil || perr != nil {
					continue
				}
				if len(grec) == 0 || len(prec) == 0 {
					continue
				}
			gs := grec[0].Confidence
			ps := prec[0].Confidence
			if len(samples) < sampleCap {
				samples = append(samples, sample{
					PDF: name, GoText: grec[0].Text, PyText: prec[0].Text,
					GoConf: gs, PyConf: ps,
				})
			}
			n++
				sumGo += gs
				sumPy += ps
				if gs >= 0.5 && ps >= 0.5 {
					bothHigh++
					goHighSum += gs
					pyHighSum += ps
				}
				if gs < 0.5 && ps >= 0.5 {
					goDropPyKeep++
				}
				if ps < 0.5 && gs >= 0.5 {
					pyDropGoKeep++
				}
			}
		}
		eng.Close()
		t.Logf("%s done", name)
	}
	meanGo, meanPy, mhGo, mhPy := 0.0, 0.0, 0.0, 0.0
	if n > 0 {
		meanGo = sumGo / float64(n)
		meanPy = sumPy / float64(n)
	}
	if bothHigh > 0 {
		mhGo = goHighSum / float64(bothHigh)
		mhPy = pyHighSum / float64(bothHigh)
	}
	t.Logf("RESULT matched=%d meanGo=%.4f meanPy=%.4f", n, meanGo, meanPy)
	t.Logf("RESULT both>=0.5: n=%d meanGo=%.4f meanPy=%.4f", bothHigh, mhGo, mhPy)
	t.Logf("RESULT goDropPyKeep=%d (Py keeps, Go drops)  pyDropGoKeep=%d (Go keeps, Py drops)",
		goDropPyKeep, pyDropGoKeep)
	for i, s := range samples {
		t.Logf("SAMPLE %d pdf=%q goConf=%.4f goText=%q pyConf=%.4f pyText=%q",
			i, s.PDF, s.GoConf, s.GoText, s.PyConf, s.PyText)
	}
}
