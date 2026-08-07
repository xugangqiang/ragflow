"""Drift gate for the DeepDoc Go port.

Re-runs the Python oracles (ref_det.py / ref_dla.py / ref_tsr.py /
ref_ocr_rec.py) on every committed fixture and compares their output against
the *pinned* golden JSON. The golden is the contract between Python (deepdoc)
and Go; this script makes Python's side comparable in CI, so a deepdoc logic
change is detected automatically instead of only when someone manually runs
run.sh. The Go side is covered separately by the integration tests
(Test*Integration), which compare Go vs the same golden.

Exit code is non-zero on any drift beyond tolerance, so a CI step using this
script fails and alerts.

Run with the deepdoc venv python (has deepdoc + onnxruntime):
  PYTHONPATH=/home/shenyushi/workspace/ragflow \\
  MODEL_DIR=/home/shenyushi/workspace/ragflow/rag/res/deepdoc \\
  /home/shenyushi/workspace/ragflow/.venv/bin/python check_drift.py

Tolerances mirror the Go integration tests:
  - det:    box count within 15% (same as TestNativeOCRDetectMultiPage)
  - dla/tsr: nearest-center-per-class coordinate match (2.0px / 0.05 score),
            same as compareBoxes in native_integration_test.go
  - ocr-rec: exact recognized text (same as TestOCRRecIntegration)
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "testdata")
MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc")

COORD_TOL = 2.0
SCORE_TOL = 0.05


def run_ref(script, stem):
    """Run a ref_*.py script on <stem>.png and return parsed JSON output."""
    img = os.path.join(TESTDATA, stem + ".png")
    if not os.path.exists(img):
        raise FileNotFoundError(img)
    cmd = [sys.executable, os.path.join(HERE, script), img, MODEL_DIR]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       env={**os.environ, "MODEL_DIR": MODEL_DIR})
    if p.returncode != 0:
        raise RuntimeError(f"{script} failed on {stem}: {p.stderr.strip()}")
    return json.loads(p.stdout)


def load_golden(path):
    with open(path) as f:
        return json.load(f)


def compare_boxes(gold, got, coord_tol=COORD_TOL, score_tol=SCORE_TOL):
    """Nearest-center-per-class match, mirroring Go's compareBoxes."""
    used = [False] * len(got)
    for gb in gold:
        cls = int(gb[5])
        bcx, bcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
        best, bd = -1, float("inf")
        for i, vb in enumerate(got):
            if used[i] or int(vb[5]) != cls:
                continue
            vcx, vcy = (vb[0] + vb[2]) / 2, (vb[1] + vb[3]) / 2
            d = (bcx - vcx) ** 2 + (bcy - vcy) ** 2
            if d < bd:
                bd, best = d, i
        if best < 0:
            return False, f"no Go box matched golden class {cls}"
        used[best] = True
        for j in range(6):
            tol = coord_tol if j != 4 else score_tol
            if abs(gb[j] - got[best][j]) > tol:
                return False, f"class {cls} coord {j} diff {abs(gb[j]-got[best][j]):.3f} > {tol}"
    return True, "ok"


def _as_boxes(obj):
    """Normalize a golden/ref payload to a list of boxes.

    DLA/TSR payloads are top-level JSON lists: [[x0,y0,x1,y1,score,class], ...].
    det-style payloads nest under {"output": [[[quads]]]}. Some legacy
    fixtures used {"bboxes": [...]}. Accept all three shapes.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "bboxes" in obj:
            return obj["bboxes"]
        out = obj.get("output")
        if out:
            # det nests quads under output[page][?]; collapse to the box list.
            return out[0][0] if out[0] else []
    return []


def check_det(stem, gold):
    got = run_ref("ref_det.py", stem)
    g = len(_as_boxes(gold))
    n = len(_as_boxes(got))
    tol = 2
    if d := int(0.15 * g):
        tol = d
    if abs(n - g) > tol:
        return False, f"box count {n} vs golden {g} (tol {tol})"
    return True, f"count {n}/{g}"


def check_bboxes(stem, gold, script):
    got = run_ref(script, stem)
    g = _as_boxes(gold)
    n = _as_boxes(got)
    if not g:
        return False, "golden has no boxes"
    ok, msg = compare_boxes(g, n)
    if not ok:
        return False, msg
    return True, f"{len(n)}/{len(g)} boxes"


def check_ocr(stem, gold):
    got = run_ref("ref_ocr_rec.py", stem)
    # Wire format: output[batch][page][items][pair] -> [text, score]; the text
    # is the string at output[0][0][0][0] (4 levels), not a 5th index which
    # would slice into the first character.
    gt = gold["output"][0][0][0][0]
    nt = got["output"][0][0][0][0]
    if gt != nt:
        return False, f"text mismatch: got {nt!r} gold {gt!r}"
    return True, f"text {nt!r}"


# (golden suffix, ref script, compare fn)
KINDS = [
    (".det.golden.json", "ref_det.py", check_det),
    (".dla.golden.json", "ref_dla.py", lambda s, g: check_bboxes(s, g, "ref_dla.py")),
    (".tsr.golden.json", "ref_tsr.py", lambda s, g: check_bboxes(s, g, "ref_tsr.py")),
    (".ocr_rec.golden.json", "ref_ocr_rec.py", check_ocr),
]


def main():
    failures = []
    checked = 0
    for fn in sorted(os.listdir(TESTDATA)):
        for suffix, script, cmpfn in KINDS:
            if not fn.endswith(suffix):
                continue
            stem = fn[: -len(suffix)]
            gold = load_golden(os.path.join(TESTDATA, fn))
            try:
                ok, msg = cmpfn(stem, gold)
            except Exception as e:  # noqa: BLE001 - surface as drift in CI
                failures.append(f"{stem}: ERROR {e}")
                continue
            checked += 1
            status = "OK " if ok else "DRIFT"
            print(f"[{status}] {stem} ({suffix.strip('.golden.json')}): {msg}")
            if not ok:
                failures.append(f"{stem}: {msg}")

    print(f"\nchecked {checked} fixtures, {len(failures)} drift(s)")
    if failures:
        print("DRIFT DETECTED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("no drift")


if __name__ == "__main__":
    main()
