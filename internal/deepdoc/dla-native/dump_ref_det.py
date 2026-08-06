"""Dump deepdoc TextDetector's exact ONNX input blob and output pred for a
given image, so they can be compared byte-for-byte against the Go port's
DLA_DUMP_PRED dumps.

This replicates exactly what ref_det.py feeds deepdoc (RGB PIL image ->
TextDetector.preprocess_op -> expand_dims -> session.run), so the blob here
is byte-for-byte what production deepdoc feeds onnxruntime.

Writes:
  /tmp/ref_blob.bin   float32 LE, layout [1,3,rh,rw] flattened
  /tmp/ref_blob_dims.txt  "rh rw sh sw"
  /tmp/ref_pred.bin   float32 LE, layout [rh,rw] flattened (sigmoid_0.tmp_0)
"""
import os
import sys
import numpy as np
from PIL import Image
from deepdoc.vision.ocr import transform
from deepdoc.vision.operators import (
    DetResizeForTest,
    NormalizeImage,
    ToCHWImage,
    KeepKeys,
)
from deepdoc.vision.ocr import create_operators, load_model

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/shenyushi/workspace/ragflow/rag/res/deepdoc")


def main():
    img_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else MODEL_DIR

    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]

    # Replicate TextDetector.__call__ preprocess exactly.
    pre_process_list = [
        {"DetResizeForTest": {"limit_side_len": 960, "limit_type": "max"}},
        {"NormalizeImage": {"std": [0.229, 0.224, 0.225],
                            "mean": [0.485, 0.456, 0.406],
                            "scale": "1./255.", "order": "hwc"}},
        {"ToCHWImage": None},
        {"KeepKeys": {"keep_keys": ["image", "shape"]}},
    ]
    ops = create_operators(pre_process_list)
    data = {"image": img}
    data = transform(data, ops)
    blob_chw, shape_list = data  # [3,rh,rw]
    blob = np.expand_dims(blob_chw, axis=0).astype("float32").copy()  # [1,3,rh,rw]
    rh, rw = blob_chw.shape[1], blob_chw.shape[2]

    # Dump the input blob (matches Go layout: flattened [1,3,rh,rw]).
    with open("/tmp/ref_blob.bin", "wb") as f:
        f.write(blob.reshape(-1).astype("<f4").tobytes())
    with open("/tmp/ref_blob_dims.txt", "w") as f:
        f.write("%d %d %d %d" % (rh, rw, h, w))

    # Run the model exactly like TextDetector.
    sess, run_options = load_model(model_dir, "det", 0)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: blob}, run_options)
    pred = outputs[0]  # [1,1,rh,rw]

    # Dump output (flatten [rh,rw], matching Go).
    pred_flat = pred.reshape(rh, rw).astype("<f4")
    with open("/tmp/ref_pred.bin", "wb") as f:
        f.write(pred_flat.tobytes())
    with open("/tmp/ref_pred_dims.txt", "w") as f:
        f.write("%d %d %d %d" % (rh, rw, h, w))

    print("blob shape", blob.shape, "pred shape", pred.shape,
          "pred min/max", float(pred.min()), float(pred.max()))


if __name__ == "__main__":
    main()
