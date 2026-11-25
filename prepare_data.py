"""Convert 300W, WFLW, AFLW2000-3D into one annotation file (68 points, original-image pixels)."""
import argparse, glob, json, os
import numpy as np

# WFLW 98 -> 68 (verify visually with dataset.py --viz!)
W98_TO_68 = (list(range(0, 33, 2)) + [33, 34, 35, 36, 37] + [42, 43, 44, 45, 46] +
             [51, 52, 53, 54] + [55, 56, 57, 58, 59] + [60, 61, 63, 64, 65, 67] +
             [68, 69, 71, 72, 73, 75] + list(range(76, 88)) + list(range(88, 96)))
assert len(W98_TO_68) == 68


def read_pts(path):
    lines = [l.strip() for l in open(path).read().strip().splitlines()]
    a, b = lines.index("{") + 1, lines.index("}")
    return np.array([[float(v) for v in l.split()] for l in lines[a:b]]) - 1.0  # 1-based -> 0-based


def do_300w(root):
    out = []
    for p in glob.glob(os.path.join(root, "**", "*.pts"), recursive=True):
        base = os.path.splitext(p)[0]
        img = next((base + e for e in (".png", ".jpg", ".jpeg", ".PNG", ".JPG") if os.path.exists(base + e)), None)
        if img is None:
            continue
        pts = read_pts(p)
        if pts.shape != (68, 2):
            continue
        low = p.replace("\\", "/").lower()
        val = any(k in low for k in ("/ibug/", "testset", "01_indoor", "02_outdoor"))
        out.append(dict(img=os.path.abspath(img), pts=pts.tolist(), split="val" if val else "train", src="300w"))
    return out


def do_wflw(root):
    ann = os.path.join(root, "WFLW_annotations", "list_98pt_rect_attr_train_test")
    out = []
    for split, fn in (("train", "list_98pt_rect_attr_train.txt"), ("val", "list_98pt_rect_attr_test.txt")):
        for line in open(os.path.join(ann, fn)):
            f = line.split()
            if len(f) < 207:
                continue
            pts = np.array(f[:196], float).reshape(98, 2)[W98_TO_68]
            img = os.path.abspath(os.path.join(root, "WFLW_images", f[206]))
            out.append(dict(img=img, pts=pts.tolist(), split=split, src="wflw"))
    return out


def do_aflw(root):
    import scipy.io as sio
    out = []
    for m in glob.glob(os.path.join(root, "**", "*.mat"), recursive=True):
        d = sio.loadmat(m)
        img = os.path.splitext(m)[0] + ".jpg"
        if "pt2d" not in d or not os.path.exists(img):
            continue
        item = dict(img=os.path.abspath(img), pts=d["pt2d"].T.tolist(), split="aflw", src="aflw2000")
        if "Pose_Para" in d:  # pitch, yaw, roll (radians)
            item["pose"] = d["Pose_Para"].reshape(-1)[:3].tolist()
        out.append(item)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w300"); ap.add_argument("--wflw"); ap.add_argument("--aflw")
    ap.add_argument("--out", default="data/annotations.json")
    a = ap.parse_args()
    items = []
    if a.w300: items += do_300w(a.w300)
    if a.wflw: items += do_wflw(a.wflw)
    if a.aflw: items += do_aflw(a.aflw)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(items, open(a.out, "w"))
    from collections import Counter
    print(Counter((d["src"], d["split"]) for d in items))