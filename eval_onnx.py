"""Compare NME of float vs INT8 ONNX on a split (val or aflw)."""
import argparse
import numpy as np, onnxruntime as ort, torch
from common import nme_per_point, REGIONS
from dataset import FaceLM

ap = argparse.ArgumentParser()
ap.add_argument("--ann", default="data/annotations.json")
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()
ds = FaceLM(a.ann, a.split, False)
n = min(a.n, len(ds))
for path in a.models:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    errs = []
    for i in range(n):
        x, y = ds[i]
        out = sess.run(None, {"input": x.numpy()[None]})[0]
        errs.append(nme_per_point(torch.from_numpy(out), y[None]))
    e = torch.cat(errs); nme = e.mean(1)
    print(f"{path}: NME {100*nme.mean():.2f}%  fail(>8%) {100*(nme>.08).float().mean():.2f}%  " +
          " ".join(f"{k} {100*e[:, s:t].mean():.2f}" for k, (s, t) in REGIONS.items()))