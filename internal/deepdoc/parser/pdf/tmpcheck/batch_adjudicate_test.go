package tmpcheck

import (
	"context"
	"encoding/json"
	"fmt"
	"native"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	pdfpkg "ragflow/internal/deepdoc/parser/pdf"
	pdftype "ragflow/internal/deepdoc/parser/pdf/type"
	infnative "ragflow/internal/deepdoc/parser/pdf/inference/native_analyzer"
	inf "ragflow/internal/deepdoc/parser/pdf/inference"
)

var secRe = regexp.MustCompile(`^\[s(\d+)\.l(\d+)\]\s?(.*)$`)

type lineRec struct {
	page int
	sec  int
	text string // without [s..] prefix
}

type blockDiff struct {
	Kind  string   `json:"kind"` // content|wrap|goOnly|pyOnly
	Page  int      `json:"page"`
	Go    []string `json:"go"`
	Py    []string `json:"py"`
}

type pdfDiffReport struct {
	PDF          string       `json:"pdf"`
	GoSections   int          `json:"go_sections"`
	PySections   int          `json:"py_sections"`
	GoLines      int          `json:"go_lines"`
	PyLines      int          `json:"py_lines"`
	Blocks       []blockDiff  `json:"blocks"`
	NumContent   int          `json:"num_content_blocks"`
	NumNumeric   int          `json:"num_numeric_mismatch"`
	NumericCells []blockDiff  `json:"numeric_cells"`
}

func loadLines(path string) []lineRec {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []lineRec
	for _, raw := range strings.Split(string(data), "\n") {
		if raw == "" {
			continue
		}
		tab := strings.Index(raw, "\t")
		if tab < 0 {
			continue
		}
		page, rest := raw[:tab], raw[tab+1:]
		p := -1
		fmt.Sscanf(page, "p%d", &p)
		m := secRe.FindStringSubmatch(rest)
		txt := rest
		sec := -1
		if m != nil {
			fmt.Sscanf(m[1], "%d", &sec)
			txt = m[3]
		}
		out = append(out, lineRec{page: p, sec: sec, text: txt})
	}
	return out
}

func normKey(t string) string {
	return strings.Join(strings.Fields(t), "")
}

func classifyDiff(g, p string) string {
	if g == "" {
		return "pyOnly"
	}
	if p == "" {
		return "goOnly"
	}
	if normKey(g) == normKey(p) {
		return "wrap"
	}
	return "content"
}

func hasDigit(s string) bool {
	for _, r := range s {
		if r >= '0' && r <= '9' {
			return true
		}
	}
	return false
}

func TestBatchAdjudicate(t *testing.T) {
	pdfDir := os.Getenv("INPROC_PDF_DIR")
	if pdfDir == "" {
		pdfDir = filepath.Join("testdata", "real_pdfs")
	}
	outDir := os.Getenv("ADJUDICATE_OUT")
	if outDir == "" {
		outDir = filepath.Join("testdata", "output", "render_compare", "adjudicate")
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
		entries, err := os.ReadDir(pdfDir)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".pdf") {
				continue
			}
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)

	pageCap := 0
	if v := os.Getenv("PAGE_CAP"); v != "" {
		fmt.Sscanf(v, "%d", &pageCap)
	}

	ctx := context.Background()
	for _, name := range names {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Logf("%s: PANIC recovered: %v", name, r)
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
			parser := pdfpkg.NewParser(pdftype.DefaultParserConfig())
			goRes, err := parser.ParseRaw(ctx, eng, goAna)
			if err != nil {
				t.Logf("%s: go parse: %v", name, err)
				return
			}
			pyRes, err := parser.ParseRaw(ctx, eng, pyAna)
			if err != nil {
				t.Logf("%s: py parse: %v", name, err)
				return
			}

			goLines := flatLines(goRes, pageCap)
			pyLines := flatLines(pyRes, pageCap)

			// Dump raw for offline diff.
			dumpLines(filepath.Join(outDir, base(name)+"_go.tsv"), goLines)
			dumpLines(filepath.Join(outDir, base(name)+"_py.tsv"), pyLines)

			blocks := diffLines(goLines, pyLines)
			rep := pdfDiffReport{
				PDF:        name,
				GoSections: len(goRes.Sections),
				PySections: len(pyRes.Sections),
				GoLines:    len(goLines),
				PyLines:    len(pyLines),
			}
			for _, b := range blocks {
				switch b.Kind {
				case "content":
					rep.NumContent++
					rep.Blocks = append(rep.Blocks, b)
					// numeric mismatch if any side has digits and values differ
					if hasDigit(strings.Join(b.Go, " ")) || hasDigit(strings.Join(b.Py, " ")) {
						rep.NumNumeric++
						rep.NumericCells = append(rep.NumericCells, b)
					}
				case "goOnly", "pyOnly":
					rep.Blocks = append(rep.Blocks, b)
				}
			}
			jb, _ := json.MarshalIndent(rep, "", "  ")
			_ = os.WriteFile(filepath.Join(outDir, base(name)+"_report.json"), jb, 0o644)
			fmt.Printf("%s: goSec=%d pySec=%d content=%d numeric=%d blocks=%d\n",
				name, rep.GoSections, rep.PySections, rep.NumContent, rep.NumNumeric, len(rep.Blocks))
		}()
	}
	fmt.Printf("DONE. reports in %s\n", outDir)
}

func base(name string) string {
	return strings.TrimSuffix(name, ".pdf")
}

func flatLines(res *pdftype.ParseResult, pageCap int) []lineRec {
	var out []lineRec
	for si, s := range res.Sections {
		page := -1
		if len(s.Positions) > 0 && len(s.Positions[0].PageNumbers) > 0 {
			page = s.Positions[0].PageNumbers[0]
		}
		if pageCap > 0 && page >= pageCap {
			continue
		}
		for li, ln := range strings.Split(s.Text, "\n") {
			out = append(out, lineRec{page: page, sec: si, text: fmt.Sprintf("[s%d.l%d] %s", si, li, ln)})
		}
	}
	return out
}

func dumpLines(path string, lines []lineRec) {
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()
	for _, l := range lines {
		fmt.Fprintf(f, "p%d\t%s\n", l.page, l.text)
	}
}

type hunk struct {
	kind  string
	page  int
	goTxt []string
	pyTxt []string
}

func diffLines(goL, pyL []lineRec) []blockDiff {
	n, m := len(goL), len(pyL)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}
	gk := make([]string, n)
	for i := range goL {
		gk[i] = normKey(goL[i].text)
	}
	pk := make([]string, m)
	for j := range pyL {
		pk[j] = normKey(pyL[j].text)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if gk[i] == pk[j] {
				dp[i][j] = dp[i+1][j+1] + 1
			} else if dp[i+1][j] >= dp[i][j+1] {
				dp[i][j] = dp[i+1][j]
			} else {
				dp[i][j] = dp[i][j+1]
			}
		}
	}
	var blocks []blockDiff
	i, j := 0, 0
	var cur *hunk
	flush := func() {
		if cur != nil {
			b := blockDiff{Kind: cur.kind, Page: cur.page, Go: cur.goTxt, Py: cur.pyTxt}
			blocks = append(blocks, b)
			cur = nil
		}
	}
	goText := func(i int) string {
		m := secRe.FindStringSubmatch(goL[i].text)
		if m != nil {
			return m[3]
		}
		return goL[i].text
	}
	pyText := func(j int) string {
		m := secRe.FindStringSubmatch(pyL[j].text)
		if m != nil {
			return m[3]
		}
		return pyL[j].text
	}
	for i < n && j < m {
		if gk[i] == pk[j] {
			flush()
			i++
			j++
		} else if dp[i+1][j] >= dp[i][j+1] {
			if cur == nil {
				cur = &hunk{}
			}
			cur.goTxt = append(cur.goTxt, goText(i))
			if cur.page < 0 {
				cur.page = goL[i].page
			}
			i++
		} else {
			if cur == nil {
				cur = &hunk{}
			}
			cur.pyTxt = append(cur.pyTxt, pyText(j))
			if cur.page < 0 {
				cur.page = pyL[j].page
			}
			j++
		}
	}
	for ; i < n; i++ {
		if cur == nil {
			cur = &hunk{}
		}
		cur.goTxt = append(cur.goTxt, goText(i))
		if cur.page < 0 {
			cur.page = goL[i].page
		}
	}
	for ; j < m; j++ {
		if cur == nil {
			cur = &hunk{}
		}
		cur.pyTxt = append(cur.pyTxt, pyText(j))
		if cur.page < 0 {
			cur.page = pyL[j].page
		}
	}
	flush()
	// classify
	for idx := range blocks {
		g := strings.Join(blocks[idx].Go, "\n")
		p := strings.Join(blocks[idx].Py, "\n")
		blocks[idx].Kind = classifyDiff(g, p)
	}
	return blocks
}
