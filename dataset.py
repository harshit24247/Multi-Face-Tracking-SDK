"""Detector-like crop + augmentation. Run with --viz to inspect samples."""
import argparse, json, math, random
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from common import S, FLIP


def make_crop(img, pts, train, margin=1.25, rot_aug=20):
    le, re = pts[36:42].mean(0), pts[42:48].mean(0)
    ang = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))  # eye-line roll (alignment)
    mn, mx = pts.min(0), pts.max(0)
    c, size = (mn + mx) / 2, float(max(mx - mn) * margin)
    if train:  # imitate detector box error and alignment error
        ang += random.uniform(-rot_aug, rot_aug)
        c = c + np.random.uniform(-.1, .1, 2) * size
        size *= random.uniform(.85, 1.15)
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, S / size)
    M[:, 2] += [S / 2 - c[0], S / 2 - c[1]]
    crop = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return crop, pts @ M[:, :2].T + M[:, 2], M


def photo(img):
    img = img.astype(np.float32)
    img = img * random.uniform(.7, 1.3) + random.uniform(-25, 25)
    if random.random() < .3:
        img = 255 * (np.clip(img, 0, 255) / 255) ** random.uniform(.7, 1.4)
    if random.random() < .3:
        k = random.choice([3, 5]); img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < .3:
        s = int(S * random.uniform(.4, .8)); img = cv2.resize(cv2.resize(img, (s, s)), (S, S))
    if random.random() < .3:
        img = img + np.random.randn(*img.shape) * random.uniform(2, 8)
    if random.random() < .3:
        w, h = random.randint(20, 70), random.randint(20, 70)
        x, y = random.randint(0, S - w), random.randint(0, S - h)
        img[y:y + h, x:x + w] = np.random.randint(0, 255, 3)
    if random.random() < .15:
        img = np.repeat(img.mean(2, keepdims=True), 3, 2)
    return np.clip(img, 0, 255)


class FaceLM(Dataset):
    def __init__(self, ann, split, train):
        self.items = [d for d in json.load(open(ann)) if d["split"] == split]
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        img = cv2.imread(d["img"], cv2.IMREAD_COLOR)
        if img is None:
            return self[random.randrange(len(self))]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pts = np.array(d["pts"], np.float32)
        if self.train and random.random() < .5:  # flip WITH left/right index swap
            img = img[:, ::-1]
            pts[:, 0] = img.shape[1] - 1 - pts[:, 0]
            pts = pts[FLIP]
        crop, p, _ = make_crop(np.ascontiguousarray(img), pts, self.train)
        if self.train:
            crop = photo(crop)
        x = (crop.astype(np.float32) - 127.5) / 127.5  # same normalisation as firmware
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy((p / S).astype(np.float32).reshape(-1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/annotations.json")
    ap.add_argument("--split", default="train")
    ap.add_argument("--viz", default="viz.jpg")
    a = ap.parse_args()
    ds = FaceLM(a.ann, a.split, a.split == "train")
    tiles = []
    for i in random.sample(range(len(ds)), 16):
        x, y = ds[i]
        im = np.ascontiguousarray((x.numpy().transpose(1, 2, 0) * 127.5 + 127.5).clip(0, 255).astype(np.uint8))
        for j, (px, py) in enumerate((y.numpy().reshape(68, 2) * S)):
            cv2.circle(im, (int(px), int(py)), 2, (255, 0, 0) if j < 17 else (0, 255, 0), -1)
        tiles.append(im)
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])
    cv2.imwrite(a.viz, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("saved", a.viz)